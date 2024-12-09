from flask import Flask, jsonify, request, current_app as app
import requests
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import util
import numpy as np
from datetime import datetime, time
from flask_cors import CORS
import torch
from modules.keyword_matching import find_top_matches, key_value_pairs
from modules.data_fetching import (fetch_users, fetch_activities, fetch_subcategories, fetch_user_subcategories, fetch_user_feedbacks, get_engine)
from modules.caching import get_cached_profile
from modules.embedding import compute_semantic_similarity, compute_tfidf_similarity, model, device
from modules.similarity import calculate_distance, calculate_age
from modules.feedback import (get_feedback_subcategories, adjust_similarity_with_feedback, adjust_profile_with_feedback)
from modules.vector_search import VectorSearch
from config import NET_CORE_API_BASE_URL
import random

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

app = Flask(__name__)
CORS(app)

def generate_user_profile(user_id, user_subcategories_df, subcategories_df):
    user_subcategories = user_subcategories_df[user_subcategories_df['UserId'] == user_id]
    if user_subcategories.empty:
        return ''
    subcategory_names = []
    for _, row in user_subcategories.iterrows():
        subcategory_row = subcategories_df[subcategories_df['Id'] == row['SubcategoryId']]
        if not subcategory_row.empty:
            subcategory_name = subcategory_row['Name'].values[0]
            subcategory_names.append(subcategory_name)
    return ' '.join(subcategory_names)

def generate_activity_profile(activity_id, activities_df):
    activity = activities_df[activities_df['Id'] == activity_id]
    if not activity.empty:
        return activity['Description'].values[0].strip()
    return ''

def precompute_data():
    print("DEBUG: Precomputing data at startup...")

    # Fetch data from DB
    app.cached_users_df = fetch_users()
    app.cached_activities_df = fetch_activities()
    app.cached_subcategories_df = fetch_subcategories()
    app.cached_user_subcategories_df = fetch_user_subcategories()

    engine = get_engine()
    user_unavailabilities_df = pd.read_sql("SELECT UserId, DayOfWeek, StartTime, EndTime FROM UserUnavailabilities", engine)

    users_df = app.cached_users_df
    activities_df = app.cached_activities_df
    subcategories_df = app.cached_subcategories_df
    user_subcategories_df = app.cached_user_subcategories_df

    # Precompute user profiles & embeddings
    app.user_profiles = {}
    all_user_ids = []
    all_user_profiles = []

    for _, user in users_df.iterrows():
        user_id = user['Id']
        user_profile = generate_user_profile(user_id, user_subcategories_df, subcategories_df)
        app.user_profiles[user_id] = user_profile
        all_user_ids.append(user_id)
        all_user_profiles.append(user_profile)

    user_embeddings_tensor = model.encode(all_user_profiles, convert_to_tensor=True).to(device)
    user_embeddings_list = user_embeddings_tensor.cpu().numpy().tolist()
    app.user_embeddings = {uid: emb for uid, emb in zip(all_user_ids, user_embeddings_list)}

    # Insert user embeddings into Milvus
    app.vector_search = VectorSearch(collection_name="user_embeddings", dim=user_embeddings_tensor.shape[1])
    app.vector_search.insert_user_embeddings(all_user_ids, user_embeddings_list)

    # Precompute activity profiles & embeddings
    app.activity_profiles = {}
    all_activity_ids = []
    all_activity_profiles = []

    for _, activity in activities_df.iterrows():
        activity_id = activity['Id']
        a_profile = generate_activity_profile(activity_id, activities_df)
        app.activity_profiles[activity_id] = a_profile
        all_activity_ids.append(activity_id)
        all_activity_profiles.append(a_profile)

    activity_embeddings_tensor = model.encode(all_activity_profiles, convert_to_tensor=True).to(device)
    activity_embeddings_list = activity_embeddings_tensor.cpu().numpy().tolist()

    app.activity_embeddings = {aid: emb for aid, emb in zip(all_activity_ids, activity_embeddings_list)}

    # Insert activity embeddings into Milvus
    app.vector_search_activities = VectorSearch(collection_name="activity_embeddings", dim=activity_embeddings_tensor.shape[1])
    app.vector_search_activities.insert_user_embeddings(all_activity_ids, activity_embeddings_list)  

    # Precompute user unavailabilities
    app.user_unavailabilities = {}
    for _, row in user_unavailabilities_df.iterrows():
        uid = row['UserId']
        day = row['DayOfWeek']
        start = row['StartTime']
        end = row['EndTime']
        if uid not in app.user_unavailabilities:
            app.user_unavailabilities[uid] = {}
        if day not in app.user_unavailabilities[uid]:
            app.user_unavailabilities[uid][day] = []
        app.user_unavailabilities[uid][day].append((start, end))

    print("DEBUG: Precomputation completed.")

def is_user_available(user_id, activity_datetime):
    day_of_week = activity_datetime.weekday() 
    time_of_day = activity_datetime.time()

    if user_id in app.user_unavailabilities:
        if day_of_week in app.user_unavailabilities[user_id]:
            blocks = app.user_unavailabilities[user_id][day_of_week]
            for (start, end) in blocks:
                if start <= time_of_day <= end:
                    return False
    return True

def explain_recommendation(
    user_id, 
    recommended_user, 
    user_subcat_list, 
    positive_subcategories, 
    requester_own_subcategories,
    distance, 
    semantic_similarity, 
    tfidf_similarity_activity, 
    tfidf_similarity_owner, 
    age_similarity, 
    subcategories_df
):
    intros = [
        "We think you’ll hit it off because",
        "Here’s why we believe you’d connect:",
        "Based on what we know, you’re a great match because"
    ]
    closing_lines = [
        "We hope you enjoy meeting this recommended user!",
        "Sounds like a match worth exploring!",
        "Why not say hello and see what happens?"
    ]

    insights = []

    # Distinguish between direct matches and feedback-inferred matches
    direct_matches = set(user_subcat_list).intersection(requester_own_subcategories)
    feedback_matches = set(user_subcat_list).intersection(positive_subcategories)

    # Convert subcategory IDs to names
    def subs_to_names(sub_ids):
        return [subcategories_df[subcategories_df['Id'] == sid]['Name'].values[0]
                for sid in sub_ids if not subcategories_df[subcategories_df['Id'] == sid].empty]

    direct_names = subs_to_names(direct_matches)
    feedback_names = subs_to_names(feedback_matches)

    # Mention direct chosen subcategories if any
    if direct_names:
        if len(direct_names) == 1:
            insights.append(f"you both share a personal favorite: {direct_names[0]}")
        else:
            insights.append(f"you share interests you personally chose, like {', '.join(direct_names[:3])}")

    # Mention feedback-inferred subcategories if any
    if feedback_names:
        if len(feedback_names) == 1:
            insights.append(f"you both connect on {feedback_names[0]}, something you grew to like through positive feedback")
        else:
            insights.append(f"you also align on interests you liked via feedback, such as {', '.join(feedback_names[:3])}")

    # Distance
    if distance < 5:
        insights.append("you’re practically neighbors!")
    elif distance < 20:
        insights.append("you live relatively close by")
    else:
        insights.append("distance won’t stop a good connection")

    # Semantic similarity
    if semantic_similarity > 0.7:
        insights.append("your overall interests align remarkably well")
    elif semantic_similarity > 0.4:
        insights.append("there’s enough overlap in your interests to spark interesting chats")

    # TF-IDF similarities
    if tfidf_similarity_activity > 0.4:
        insights.append("you share common ground related to this activity's theme")
    if tfidf_similarity_owner > 0.4:
        insights.append("your tastes harmonize with the activity creator’s style")

    # Age similarity
    if age_similarity > 0.5:
        insights.append("you’re around the same age, making it easy to relate")

    if not insights:
        insights.append("you share some common interests")

    import random
    introduction = random.choice(intros)
    closing = random.choice(closing_lines)

    if len(insights) > 1:
        narrative = introduction + " " + "; ".join(insights[:-1]) + " and " + insights[-1] + ". " + closing
    else:
        narrative = introduction + " " + insights[0] + ". " + closing

    return narrative

def explain_activity_recommendation(user_id, activity_id, user_subcat_list, positive_subcategories, requester_own_subcategories,
                                   distance, semantic_similarity, tfidf_similarity, age_similarity, subcategories_df):
    intros = [
        "We think you’ll enjoy this activity because",
        "Here’s why we believe you’d be interested in this activity:",
        "Based on what we know, this activity might suit you because"
    ]
    closing_lines = [
        "Why not give it a try?",
        "Sounds like a great opportunity!",
        "We hope you find this activity exciting!"
    ]

    insights = []

    # Distinguish between direct chosen subcategories and feedback-inferred matches.
    direct_matches = set(user_subcat_list).intersection(requester_own_subcategories)
    feedback_matches = set(user_subcat_list).intersection(positive_subcategories)

    def subs_to_names(sub_ids):
        return [subcategories_df[subcategories_df['Id'] == sid]['Name'].values[0]
                for sid in sub_ids if not subcategories_df[subcategories_df['Id'] == sid].empty]

    direct_names = subs_to_names(direct_matches)
    feedback_names = subs_to_names(feedback_matches)

    # Mention direct chosen subcategories
    if direct_names:
        if len(direct_names) == 1:
            insights.append(f"it matches your personal favorite interest: {direct_names[0]}")
        else:
            insights.append(f"it aligns with interests you personally chose, like {', '.join(direct_names[:3])}")

    # Mention feedback-inferred subcategories
    if feedback_names:
        if len(feedback_names) == 1:
            insights.append(f"it touches on {feedback_names[0]}, an interest you liked based on previous feedback")
        else:
            insights.append(f"it resonates with interests you grew to like, such as {', '.join(feedback_names[:3])}")

    # Distance factor
    if distance < 5:
        insights.append("the location is right around the corner")
    elif distance < 20:
        insights.append("it's not too far from you")
    else:
        insights.append("even if it's a bit farther away, good experiences can be worth the travel")

    # Semantic similarity
    if semantic_similarity > 0.7:
        insights.append("its overall theme aligns remarkably well with your profile")
    elif semantic_similarity > 0.4:
        insights.append("there’s enough overlap in what you enjoy to spark your interest")

    # TF-IDF similarity
    if tfidf_similarity > 0.4:
        insights.append("the specific details of the activity match well with your tastes")

    # Age similarity
    if age_similarity > 0.5:
        insights.append("the activity creator is around your age, making it more relatable")

    if not insights:
        insights.append("it generally matches some of your interests")

    import random
    introduction = random.choice(intros)
    closing = random.choice(closing_lines)

    if len(insights) > 1:
        narrative = introduction + " " + "; ".join(insights[:-1]) + " and " + insights[-1] + ". " + closing
    else:
        narrative = introduction + " " + insights[0] + ". " + closing

    return narrative

### INCREMENTAL UPDATE FUNCTIONS ###

def update_user_subcategories(user_id):
    engine = get_engine()
    user_subcategories_df = pd.read_sql("SELECT UserId, SubcategoryId, Weight FROM UserSubcategories", engine)
    subcategories_df = pd.read_sql("SELECT Id, Name, CategoryId FROM Subcategories", engine)

    user_profile = generate_user_profile(user_id, user_subcategories_df, subcategories_df)
    app.user_profiles[user_id] = user_profile

    embedding = None
    if user_profile.strip():
        new_emb = model.encode([user_profile], convert_to_tensor=True).to(device)
        embedding = new_emb.squeeze(0).cpu().numpy().tolist()

    if embedding is not None:
        app.user_embeddings[user_id] = embedding
        app.vector_search.upsert_user_embedding(user_id, embedding)
    else:
        if user_id in app.user_embeddings:
            del app.user_embeddings[user_id]
        app.vector_search.delete_user_embedding(user_id)

def update_user_unavailability(user_id):
    engine = get_engine()
    user_unavail_df = pd.read_sql(f"SELECT UserId, DayOfWeek, StartTime, EndTime FROM UserUnavailabilities WHERE UserId={user_id}", engine)

    if user_id in app.user_unavailabilities:
        del app.user_unavailabilities[user_id]

    for _, row in user_unavail_df.iterrows():
        uid = row['UserId']
        day = row['DayOfWeek']
        start = row['StartTime']
        end = row['EndTime']
        if uid not in app.user_unavailabilities:
            app.user_unavailabilities[uid] = {}
        if day not in app.user_unavailabilities[uid]:
            app.user_unavailabilities[uid][day] = []
        app.user_unavailabilities[uid][day].append((start, end))

def add_new_activity(activity_id):
    engine = get_engine()
    new_activity_df = pd.read_sql(f"""
        SELECT Activities.Id, Activities.Name, Activities.Description, Activities.LocationId, Activities.Date, Activities.CreatedById, Locations.Latitude, Locations.Longitude
        FROM Activities
        LEFT JOIN Locations ON Activities.LocationId=Locations.Id
        WHERE Activities.Id = {activity_id}
    """, engine)

    if new_activity_df.empty:
        return

    # Append the new activity to cached_activities_df
    app.cached_activities_df = pd.concat([app.cached_activities_df, new_activity_df], ignore_index=True)

    activity_profile = generate_activity_profile(activity_id, app.cached_activities_df)
    app.activity_profiles[activity_id] = activity_profile

    activity_embedding = None
    if activity_profile.strip():
        emb = model.encode([activity_profile], convert_to_tensor=True).to(device)
        activity_embedding = emb.squeeze(0).cpu().numpy().tolist()

    app.activity_embeddings[activity_id] = activity_embedding
    if activity_embedding is not None:
        app.vector_search_activities.upsert_user_embedding(activity_id, activity_embedding)
    else:
        app.vector_search_activities.delete_user_embedding(activity_id)


### API ENDPOINTS FOR INCREMENTAL UPDATES ###

@app.route('/update_user_subcategories/<int:user_id>', methods=['POST'])
def update_user_subcategories_route(user_id):
    update_user_subcategories(user_id)
    return jsonify({"message": "User subcategories and embeddings updated"}), 200

@app.route('/update_user_unavailability/<int:user_id>', methods=['POST'])
def update_user_unavailability_route(user_id):
    update_user_unavailability(user_id)
    return jsonify({"message": "User unavailabilities updated"}), 200

@app.route('/add_new_activity/<int:activity_id>', methods=['POST'])
def add_new_activity_route(activity_id):
    add_new_activity(activity_id)
    return jsonify({"message": "New activity added and embeddings updated"}), 200

@app.route('/refresh_data', methods=['POST'])
def refresh_data():
    if hasattr(app, 'cached_users_df'):
        del app.cached_users_df
    if hasattr(app, 'cached_activities_df'):
        del app.cached_activities_df
    if hasattr(app, 'cached_subcategories_df'):
        del app.cached_subcategories_df
    if hasattr(app, 'cached_user_subcategories_df'):
        del app.cached_user_subcategories_df

    precompute_data()
    return jsonify({"message": "Data refreshed successfully"}), 200

@app.route('/find_matches', methods=['POST'])
def find_matches():
    data = request.json
    user_input = data.get('user_input')
    if not user_input:
        return jsonify({'error': 'user_input is required'}), 400

    top_matches = find_top_matches(user_input, key_value_pairs, top_n=20, key_weight=0.3, value_weight=0.7)
    response = [{'key': k, 'value': v, 'similarity': s} for s, k, v in top_matches]
    return jsonify({'top_matches': response})

@app.route('/recommend_users', methods=['GET'])
def recommend_users():
    activity_id = request.args.get('activity_id', None)
    top_n = int(request.args.get('top_n', 20))
    # New: userId parameter
    requester_id = request.args.get('user_id', None)

    if not activity_id:
        return jsonify({"error": "activity_id is required"}), 400
    try:
        activity_id = int(activity_id)
    except ValueError:
        return jsonify({"error": "Invalid activity_id"}), 400

    if requester_id is not None:
        try:
            requester_id = int(requester_id)
        except ValueError:
            return jsonify({"error": "Invalid user_id"}), 400

    users_df = app.cached_users_df
    activities_df = app.cached_activities_df
    subcategories_df = app.cached_subcategories_df
    user_subcategories_df = app.cached_user_subcategories_df

    activity = activities_df[activities_df['Id'] == activity_id]

    # Check if activity exists
    if activity.empty:
        # Activity not found. Use userId-based recommendation if userId provided.
        if requester_id is not None and requester_id in app.user_embeddings:
            # We have a requester with embeddings, do user-to-user similarity
            requester_embedding = app.user_embeddings.get(requester_id)
            if requester_embedding is not None:
                top_candidates = app.vector_search.search_similar_users(requester_embedding, top_k=100)
                candidate_user_ids = [hit.id for hit in top_candidates]
                candidate_users = users_df[users_df['Id'].isin(candidate_user_ids)]

                requester_feedbacks = fetch_user_feedbacks(requester_id)
                positive_subcategories = get_feedback_subcategories(requester_feedbacks, user_subcategories_df, positive=True)
                requester_own_subcategories = user_subcategories_df[user_subcategories_df['UserId'] == requester_id]['SubcategoryId'].tolist()

                recommendations = []

                for _, usr in candidate_users.iterrows():
                    if usr['Id'] == requester_id:
                        continue
                    user_profile = app.user_profiles.get(usr['Id'], None)
                    if not user_profile:
                        continue

                    distance = 0.0

                    combined_similarity = 0.5
                    explanation = "We couldn’t find the specified activity, but we used your own profile to find users who share similarities with you."

                    recommendations.append({
                        'UserId': usr['Id'],
                        'UserName': usr['Name'],
                        'SimilarityScore': combined_similarity,
                        'Distance': distance,
                        'Explanation': explanation
                    })

                recommendations.sort(key=lambda x: x['SimilarityScore'], reverse=True)
                return jsonify(recommendations[:top_n])
            else:
                # requester_id does not have embeddings
                # Fallback: return basic filtered users with uniform score
                candidate_users = users_df
                recommendations = []
                for _, usr in candidate_users.iterrows():
                    if usr['Id'] == requester_id:
                        continue
                    user_profile = app.user_profiles.get(usr['Id'], None)
                    if not user_profile:
                        continue
                    recommendations.append({
                        'UserId': usr['Id'],
                        'UserName': usr['Name'],
                        'SimilarityScore': 0.5,
                        'Distance': 0.0,
                        'Explanation': "No activity and no embeddings for requester, showing a generic list of users."
                    })
                recommendations.sort(key=lambda x: x['SimilarityScore'], reverse=True)
                return jsonify(recommendations[:top_n])
        else:
            # No activity and no user_id or user_id not in embeddings: return generic list
            candidate_users = users_df
            recommendations = []
            for _, usr in candidate_users.iterrows():
                user_profile = app.user_profiles.get(usr['Id'], None)
                if not user_profile:
                    continue
                recommendations.append({
                    'UserId': usr['Id'],
                    'UserName': usr['Name'],
                    'SimilarityScore': 0.5,
                    'Distance': 0.0,
                    'Explanation': "No activity specified and no requester user_id provided, returning a generic list of users."
                })
            recommendations.sort(key=lambda x: x['SimilarityScore'], reverse=True)
            return jsonify(recommendations[:top_n])

    # If activity found, proceed as before with the activity logic:
    activity_lat = activity['Latitude'].values[0]
    activity_lon = activity['Longitude'].values[0]
    activity_profile = app.activity_profiles[activity_id]
    activity_embedding = app.activity_embeddings[activity_id]

    created_by_id = activity['CreatedById'].values[0]
    owner_row = users_df[users_df['Id'] == created_by_id]
    owner_age = None
    if not owner_row.empty and pd.notnull(owner_row['DateOfBirth'].values[0]):
        owner_age = calculate_age(owner_row['DateOfBirth'].values[0])

    owner_profile = app.user_profiles[created_by_id]
    owner_embedding = app.user_embeddings[created_by_id]

    owner_feedbacks = fetch_user_feedbacks(created_by_id)
    positive_subcategories = get_feedback_subcategories(owner_feedbacks, user_subcategories_df, positive=True)
    requester_own_subcategories = user_subcategories_df[user_subcategories_df['UserId'] == created_by_id]['SubcategoryId'].tolist()

    if activity_embedding is not None:
        query_vector = activity_embedding
        top_candidates = app.vector_search.search_similar_users(query_vector, top_k=100)
        candidate_user_ids = [hit.id for hit in top_candidates]
        candidate_users = users_df[users_df['Id'].isin(candidate_user_ids)]
    else:
        candidate_users = users_df

    recommendations = []
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit([owner_profile, activity_profile])

    activity_dt = pd.to_datetime(activity['Date'].values[0])

    for _, usr in candidate_users.iterrows():
        if usr['Id'] == created_by_id:
            continue

        distance = calculate_distance(activity_lat, activity_lon, usr['Latitude'], usr['Longitude'])
        if distance > usr['DistanceWillingToTravel']:
            continue

        if not is_user_available(usr['Id'], activity_dt):
            continue

        user_id = usr['Id']
        user_profile = app.user_profiles[user_id]
        if not user_profile:
            continue

        user_subcat_list = user_subcategories_df[user_subcategories_df['UserId'] == user_id]['SubcategoryId'].tolist()
        adjusted_user_profile = adjust_profile_with_feedback(
            user_profile, user_subcat_list, positive_subcategories, subcategories_df
        )

        tfidf_similarity_activity = compute_tfidf_similarity(adjusted_user_profile, activity_profile, tfidf_vectorizer)
        tfidf_similarity_owner = compute_tfidf_similarity(adjusted_user_profile, owner_profile, tfidf_vectorizer)

        user_embedding = app.user_embeddings.get(user_id, None)
        if user_embedding is not None and owner_embedding is not None and activity_embedding is not None:
            ue_t = torch.tensor(user_embedding, dtype=torch.float32, device=device)
            ae_t = torch.tensor(activity_embedding, dtype=torch.float32, device=device)
            oe_t = torch.tensor(owner_embedding, dtype=torch.float32, device=device)

            semantic_similarity_activity = float(util.pytorch_cos_sim(ue_t, ae_t).item())
            semantic_similarity_owner = float(util.pytorch_cos_sim(ue_t, oe_t).item())
        else:
            semantic_similarity_activity = compute_semantic_similarity(adjusted_user_profile, activity_profile)
            semantic_similarity_owner = compute_semantic_similarity(adjusted_user_profile, owner_profile)

        semantic_similarity = 0.3 * semantic_similarity_activity + 0.7 * semantic_similarity_owner

        tfidf_similarity_activity = adjust_similarity_with_feedback(
            tfidf_similarity_activity, user_subcat_list, positive_subcategories
        )

        user_age = None
        if pd.notnull(usr['DateOfBirth']):
            user_age = calculate_age(usr['DateOfBirth'])
        age_similarity = 0
        if owner_age is not None and user_age is not None:
            age_diff = abs(owner_age - user_age)
            age_similarity = max(0, 1 - (age_diff / 10))

        combined_similarity = (
            0.2 * tfidf_similarity_activity +
            0.1 * tfidf_similarity_owner +
            0.5 * semantic_similarity +
            0.2 * age_similarity
        )

        explanation = explain_recommendation(
            user_id=created_by_id,
            recommended_user=user_id,
            user_subcat_list=user_subcat_list,
            positive_subcategories=positive_subcategories,
            requester_own_subcategories=requester_own_subcategories, 
            distance=distance,
            semantic_similarity=semantic_similarity,
            tfidf_similarity_activity=tfidf_similarity_activity,
            tfidf_similarity_owner=tfidf_similarity_owner,
            age_similarity=age_similarity,
            subcategories_df=subcategories_df
        )

        recommendations.append({
            'UserId': usr['Id'],
            'UserName': usr['Name'],
            'SimilarityScore': combined_similarity,
            'Distance': distance,
            'Explanation': explanation
        })

    recommendations.sort(key=lambda x: x['SimilarityScore'], reverse=True)
    print(recommendations)

    return jsonify(recommendations[:top_n])

@app.route('/recommend_activities', methods=['GET'])
def recommend_activities():
    user_id = request.args.get('user_id', None)
    top_n = int(request.args.get('top_n', 20))

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    user_id = int(user_id)
    users_df = app.cached_users_df
    activities_df = app.cached_activities_df
    subcategories_df = app.cached_subcategories_df
    user_subcategories_df = app.cached_user_subcategories_df

    user = users_df[users_df['Id'] == user_id]
    if user.empty:
        return jsonify({"error": f"User with ID {user_id} not found"}), 404

    user_lat = user['Latitude'].values[0]
    user_lon = user['Longitude'].values[0]
    user_distance_limit = user['DistanceWillingToTravel'].values[0]

    user_profile = app.user_profiles[user_id]
    if not user_profile:
        return jsonify({"error": f"User profile could not be generated for user_id {user_id}"}), 404

    user_feedbacks = fetch_user_feedbacks(user_id)
    positive_subcategories = get_feedback_subcategories(user_feedbacks, user_subcategories_df, positive=True)

    user_age = None
    if pd.notnull(user['DateOfBirth'].values[0]):
        user_age = calculate_age(user['DateOfBirth'].values[0])

    # Vector search for similar activities
    user_embedding = app.user_embeddings.get(user_id, None)
    if user_embedding is not None:
        top_candidates = app.vector_search_activities.search_similar_users(user_embedding, top_k=100)
        candidate_activity_ids = [hit.id for hit in top_candidates]
        candidate_activities = activities_df[activities_df['Id'].isin(candidate_activity_ids)]
    else:
        # If no embedding for user, fallback to all activities
        candidate_activities = activities_df

    recommendations = []
    for _, activity in candidate_activities.iterrows():
        activity_lat = activity['Latitude']
        activity_lon = activity['Longitude']
        if pd.isnull(activity_lat) or pd.isnull(activity_lon):
            continue

        distance = calculate_distance(user_lat, user_lon, activity_lat, activity_lon)
        if distance > user_distance_limit:
            continue

        activity_profile = app.activity_profiles[activity['Id']]
        if not activity_profile:
            continue

        adjusted_user_profile = adjust_profile_with_feedback(
            user_profile,
            user_subcategories_df[user_subcategories_df['UserId'] == user_id]['SubcategoryId'].tolist(),
            positive_subcategories,
            subcategories_df
        )

        # Create and fit TF-IDF vectorizer for this specific comparison
        tfidf_vectorizer = TfidfVectorizer()
        # Fit on the two texts we're comparing
        tfidf_vectorizer.fit([adjusted_user_profile, activity_profile])

        semantic_similarity = compute_semantic_similarity(adjusted_user_profile, activity_profile)
        tfidf_similarity = compute_tfidf_similarity(adjusted_user_profile, activity_profile, tfidf_vectorizer)

        creator_id = activity['CreatedById']
        creator_age = None
        if creator_id in users_df['Id'].values:
            creator = users_df[users_df['Id'] == creator_id]
            if pd.notnull(creator['DateOfBirth'].values[0]):
                creator_age = calculate_age(creator['DateOfBirth'].values[0])

        age_similarity = 0
        if user_age is not None and creator_age is not None:
            age_difference = abs(user_age - creator_age)
            age_similarity = max(0, 1 - (age_difference / 10))

        combined_similarity = (
            0.4 * tfidf_similarity +
            0.4 * semantic_similarity +
            0.2 * age_similarity
        )

        # Just like with recommend_users, if you want explanations here:
        activity_subcats = user_subcategories_df[user_subcategories_df['UserId'] == creator_id]['SubcategoryId'].tolist()
        explanation = explain_activity_recommendation(
            user_id=user_id,
            activity_id=activity['Id'],
            user_subcat_list=activity_subcats,
            positive_subcategories=positive_subcategories,
            requester_own_subcategories=user_subcategories_df[user_subcategories_df['UserId'] == user_id]['SubcategoryId'].tolist(),
            distance=distance,
            semantic_similarity=semantic_similarity,
            tfidf_similarity=tfidf_similarity,
            age_similarity=age_similarity,
            subcategories_df=subcategories_df
        )

        recommendations.append({
            'ActivityId': activity['Id'],
            'ActivityName': activity['Name'],
            'SimilarityScore': combined_similarity,
            'Distance': distance,
            'Explanation': explanation
        })

    recommendations = sorted(recommendations, key=lambda x: x['SimilarityScore'], reverse=True)[:top_n]
    print(recommendations)
    return jsonify(recommendations)


precompute_data()

if __name__ == '__main__':
    app.run(debug=True)
