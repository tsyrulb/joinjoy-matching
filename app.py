from flask import Flask, jsonify, request, current_app as app
import requests
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import util
from multiprocessing import Pool, cpu_count
import numpy as np
from modules.keyword_matching import find_top_matches, key_value_pairs
from modules.data_fetching import (fetch_users, fetch_activities, fetch_subcategories, fetch_user_subcategories, fetch_user_feedbacks)
from modules.caching import get_cached_profile
from modules.embedding import compute_semantic_similarity, compute_tfidf_similarity
from modules.similarity import calculate_distance, calculate_age
from modules.feedback import (get_feedback_subcategories, adjust_similarity_with_feedback, adjust_profile_with_feedback)
from config import NET_CORE_API_BASE_URL
from modules.embedding import model, device
from flask_cors import CORS

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

app = Flask(__name__)
CORS(app) 

@app.route('/find_matches', methods=['POST'])
def find_matches():
    data = request.json
    user_input = data.get('user_input')

    if not user_input:
        return jsonify({'error': 'user_input is required'}), 400

    top_matches = find_top_matches(user_input, key_value_pairs, top_n=20, key_weight=0.3, value_weight=0.7)
    response = [{'key': k, 'value': v, 'similarity': s} for s, k, v in top_matches]

    return jsonify({'top_matches': response})

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

### Routes ###
def precompute_data():
    """
    This runs before the first request to the Flask app.
    Pre-fetch data from DB, generate profiles, compute embeddings,
    and store them in memory.
    """
    print("DEBUG: Precomputing data at startup...")

    # Fetch data from DB
    app.cached_users_df = fetch_users()
    app.cached_activities_df = fetch_activities()
    app.cached_subcategories_df = fetch_subcategories()
    app.cached_user_subcategories_df = fetch_user_subcategories()

    users_df = app.cached_users_df
    activities_df = app.cached_activities_df
    subcategories_df = app.cached_subcategories_df
    user_subcategories_df = app.cached_user_subcategories_df

    # Precompute user profiles and embeddings
    app.user_profiles = {}
    app.user_embeddings = {}

    for _, user in users_df.iterrows():
        user_id = user['Id']
        user_profile = generate_user_profile(user_id, user_subcategories_df, subcategories_df)
        app.user_profiles[user_id] = user_profile
        if user_profile:  # Only encode if not empty
            embedding = model.encode([user_profile], convert_to_tensor=True).to(device)
        else:
            embedding = None
        app.user_embeddings[user_id] = embedding

    # Precompute activity profiles and embeddings
    app.activity_profiles = {}
    app.activity_embeddings = {}

    for _, activity in activities_df.iterrows():
        activity_id = activity['Id']
        activity_profile = generate_activity_profile(activity_id, activities_df)
        app.activity_profiles[activity_id] = activity_profile
        if activity_profile:
            activity_embedding = model.encode([activity_profile], convert_to_tensor=True).to(device)
        else:
            activity_embedding = None
        app.activity_embeddings[activity_id] = activity_embedding

    print("DEBUG: Precomputation completed.")

@app.route('/test/matching/users', methods=['GET'])
def test_get_users():
    try:
        response = requests.get(f"{NET_CORE_API_BASE_URL}/users", verify=False)
        response.raise_for_status()
        users = response.json()
        return jsonify(users), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch users: {str(e)}"}), 500

@app.route('/test/matching/activities', methods=['GET'])
def test_get_activities():
    try:
        response = requests.get(f"{NET_CORE_API_BASE_URL}/activities", verify=False)
        response.raise_for_status()
        activities = response.json()
        return jsonify(activities), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch activities: {str(e)}"}), 500


@app.route('/recommend_users', methods=['GET'])
def recommend_users():
    activity_id = request.args.get('activity_id', None)
    top_n = int(request.args.get('top_n', 20))

    if not activity_id:
        return jsonify({"error": "activity_id is required"}), 400

    try:
        activity_id = int(activity_id)
    except ValueError:
        return jsonify({"error": "Invalid activity_id"}), 400

    # Retrieve precomputed data
    users_df = app.cached_users_df
    activities_df = app.cached_activities_df
    subcategories_df = app.cached_subcategories_df
    user_subcategories_df = app.cached_user_subcategories_df

    activity = activities_df[activities_df['Id'] == activity_id]
    if activity.empty:
        return jsonify({"error": f"Activity with ID {activity_id} not found"}), 404

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
    negative_subcategories = get_feedback_subcategories(owner_feedbacks, user_subcategories_df, positive=False)

    # Filter users by distance
    candidates = []
    for _, user in users_df.iterrows():
        if user['Id'] == created_by_id:
            continue
        distance = calculate_distance(activity_lat, activity_lon, user['Latitude'], user['Longitude'])
        if distance <= user['DistanceWillingToTravel']:
            candidates.append((user, distance))

    if not candidates:
        return jsonify([])

    recommendations = []

    # For simplicity, we still compute TF-IDF and semantic similarities per user request.
    # If this is still slow, consider also precomputing a global TF-IDF vectorizer or skipping TF-IDF.
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit([owner_profile, activity_profile])

    for (usr, dist) in candidates:
        user_id = usr['Id']
        user_profile = app.user_profiles[user_id]
        if not user_profile:
            continue

        user_subcat_list = user_subcategories_df[user_subcategories_df['UserId'] == user_id]['SubcategoryId'].tolist()

        # Adjust profile with feedback
        adjusted_user_profile = adjust_profile_with_feedback(
            user_profile,
            user_subcat_list,
            positive_subcategories,
            negative_subcategories,
            subcategories_df
        )

        # Compute similarities (TF-IDF and semantic)
        tfidf_similarity_activity = compute_tfidf_similarity(adjusted_user_profile, activity_profile, tfidf_vectorizer)
        tfidf_similarity_owner = compute_tfidf_similarity(adjusted_user_profile, owner_profile, tfidf_vectorizer)

        semantic_similarity_activity = 0
        semantic_similarity_owner = 0

        # Use precomputed embeddings if available
        user_embedding = app.user_embeddings[user_id]
        if user_embedding is not None and activity_embedding is not None:
            semantic_similarity_activity = float(util.pytorch_cos_sim(user_embedding, activity_embedding).item())
        else:
            # Fallback if any embedding is missing
            semantic_similarity_activity = compute_semantic_similarity(adjusted_user_profile, activity_profile)

        if user_embedding is not None and owner_embedding is not None:
            semantic_similarity_owner = float(util.pytorch_cos_sim(user_embedding, owner_embedding).item())
        else:
            # Fallback if any embedding is missing
            semantic_similarity_owner = compute_semantic_similarity(adjusted_user_profile, owner_profile)

        semantic_similarity = 0.3 * semantic_similarity_activity + 0.7 * semantic_similarity_owner

        # Adjust similarity with activity based on feedback
        tfidf_similarity_activity = adjust_similarity_with_feedback(
            tfidf_similarity_activity,
            user_subcat_list,
            positive_subcategories,
            negative_subcategories
        )

        # Compute age similarity
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

        recommendations.append({
            'UserId': usr['Id'],
            'UserName': usr['Name'],
            'SimilarityScore': combined_similarity,
            'Distance': dist
        })

    recommendations.sort(key=lambda x: x['SimilarityScore'], reverse=True)
    return jsonify(recommendations[:top_n])

@app.route('/recommend_activities', methods=['GET'])
def recommend_activities():
    user_id = request.args.get('user_id', None)
    top_n = int(request.args.get('top_n', 20))

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    user_id = int(user_id)
    users_df = fetch_users()
    activities_df = fetch_activities()
    subcategories_df = fetch_subcategories()
    user_subcategories_df = fetch_user_subcategories()

    user = users_df[users_df['Id'] == user_id]
    if user.empty:
        return jsonify({"error": f"User with ID {user_id} not found"}), 404

    user_lat = user['Latitude'].values[0]
    user_lon = user['Longitude'].values[0]
    user_distance_limit = user['DistanceWillingToTravel'].values[0]

    user_profile = get_cached_profile(
        f"user_profile:{user_id}",
        generate_user_profile,
        user_id,
        user_subcategories_df,
        subcategories_df
    )

    if not user_profile:
        return jsonify({"error": f"User profile could not be generated for user_id {user_id}"}), 404

    user_feedbacks = fetch_user_feedbacks(user_id)
    positive_subcategories = get_feedback_subcategories(user_feedbacks, user_subcategories_df, positive=True)
    negative_subcategories = get_feedback_subcategories(user_feedbacks, user_subcategories_df, positive=False)

    user_age = None
    if pd.notnull(user['DateOfBirth'].values[0]):
        user_age = calculate_age(user['DateOfBirth'].values[0])

    recommendations = []

    for _, activity in activities_df.iterrows():
        activity_lat = activity['Latitude']
        activity_lon = activity['Longitude']
        if pd.isnull(activity_lat) or pd.isnull(activity_lon):
            continue

        distance = calculate_distance(user_lat, user_lon, activity_lat, activity_lon)
        if distance > user_distance_limit:
            continue

        activity_profile = get_cached_profile(
            f"activity_profile:{activity['Id']}",
            generate_activity_profile,
            activity['Id'],
            activities_df
        )

        if not activity_profile:
            continue

        adjusted_user_profile = adjust_profile_with_feedback(
            user_profile,
            user_subcategories_df[user_subcategories_df['UserId'] == user_id]['SubcategoryId'].tolist(),
            positive_subcategories,
            negative_subcategories,
            subcategories_df
        )

        semantic_similarity = compute_semantic_similarity(adjusted_user_profile, activity_profile)
        tfidf_similarity = compute_tfidf_similarity(adjusted_user_profile, activity_profile)

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

        recommendations.append({
            'ActivityId': activity['Id'],
            'ActivityName': activity['Name'],
            'SimilarityScore': combined_similarity,
            'Distance': distance
        })

    recommendations = sorted(recommendations, key=lambda x: x['SimilarityScore'], reverse=True)[:top_n]
    return jsonify(recommendations)

@app.route('/refresh_data', methods=['POST'])
def refresh_data():
    # Clear cached dataframes if they exist
    if hasattr(app, 'cached_users_df'):
        del app.cached_users_df
    if hasattr(app, 'cached_activities_df'):
        del app.cached_activities_df
    if hasattr(app, 'cached_subcategories_df'):
        del app.cached_subcategories_df
    if hasattr(app, 'cached_user_subcategories_df'):
        del app.cached_user_subcategories_df

    # Re-fetch data from the database
    app.cached_users_df = fetch_users()
    app.cached_activities_df = fetch_activities()
    app.cached_subcategories_df = fetch_subcategories()
    app.cached_user_subcategories_df = fetch_user_subcategories()

    return jsonify({"message": "Data refreshed successfully"}), 200

# Precompute data
precompute_data()  

if __name__ == '__main__':
    app.run(debug=True)
