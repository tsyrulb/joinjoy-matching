from flask import Flask, jsonify, request
import requests
import warnings
import pandas as pd

from modules.keyword_matching import find_top_matches, key_value_pairs
from modules.data_fetching import (fetch_users, fetch_activities, fetch_subcategories, fetch_user_subcategories, fetch_user_feedbacks)
from modules.caching import get_cached_profile
from modules.embedding import compute_semantic_similarity, compute_tfidf_similarity
from modules.similarity import calculate_distance, calculate_age
from modules.feedback import (get_feedback_subcategories, adjust_similarity_with_feedback, adjust_profile_with_feedback)
from config import NET_CORE_API_BASE_URL

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

app = Flask(__name__)

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

    users_df = fetch_users()
    activities_df = fetch_activities()
    subcategories_df = fetch_subcategories()
    user_subcategories_df = fetch_user_subcategories()

    activity_id = int(activity_id)
    activity = activities_df[activities_df['Id'] == activity_id]
    if activity.empty:
        return jsonify({"error": f"Activity with ID {activity_id} not found"}), 404

    activity_lat = activity['Latitude'].values[0]
    activity_lon = activity['Longitude'].values[0]
    activity_profile = get_cached_profile(
        f"activity_profile:{activity_id}",
        generate_activity_profile,
        activity_id,
        activities_df
    )
    created_by_id = activity['CreatedById'].values[0]

    owner_row = users_df[users_df['Id'] == created_by_id]
    owner_age = None
    if not owner_row.empty and owner_row['DateOfBirth'].notnull().values[0]:
        owner_age = calculate_age(owner_row['DateOfBirth'].values[0])

    owner_profile = get_cached_profile(
        f"user_profile:{created_by_id}",
        generate_user_profile,
        created_by_id,
        user_subcategories_df,
        subcategories_df
    )

    owner_feedbacks = fetch_user_feedbacks(created_by_id)
    positive_subcategories = get_feedback_subcategories(owner_feedbacks, user_subcategories_df, positive=True)
    negative_subcategories = get_feedback_subcategories(owner_feedbacks, user_subcategories_df, positive=False)

    recommendations = []

    for _, user in users_df.iterrows():
        if user['Id'] == created_by_id:
            continue

        user_lat = user['Latitude']
        user_lon = user['Longitude']
        distance = calculate_distance(activity_lat, activity_lon, user_lat, user_lon)

        if distance > user['DistanceWillingToTravel']:
            continue

        user_profile = get_cached_profile(
            f"user_profile:{user['Id']}",
            generate_user_profile,
            user['Id'],
            user_subcategories_df,
            subcategories_df
        )

        if not user_profile:
            continue

        user_subcat_list = user_subcategories_df[user_subcategories_df['UserId'] == user['Id']]['SubcategoryId'].tolist()
        similarity_with_activity = compute_tfidf_similarity(user_profile, activity_profile)
        similarity_with_activity = adjust_similarity_with_feedback(
            similarity_with_activity, 
            user_subcat_list, 
            positive_subcategories, 
            negative_subcategories
        )

        adjusted_user_profile = adjust_profile_with_feedback(
            user_profile, 
            user_subcat_list, 
            positive_subcategories, 
            negative_subcategories,
            subcategories_df
        )

        semantic_similarity = compute_semantic_similarity(adjusted_user_profile, activity_profile)
        semantic_similarity_user = compute_semantic_similarity(adjusted_user_profile, owner_profile)
        semantic_similarity = 0.3 * semantic_similarity + 0.7 * semantic_similarity_user

        similarity_with_owner = compute_tfidf_similarity(user_profile, owner_profile)

        user_age = None
        if pd.notnull(user['DateOfBirth']):
            user_age = calculate_age(user['DateOfBirth'])

        age_similarity = 0
        if owner_age is not None and user_age is not None:
            age_difference = abs(owner_age - user_age)
            age_similarity = max(0, 1 - (age_difference / 10))

        combined_similarity = (
            0.2 * similarity_with_activity +
            0.1 * similarity_with_owner +
            0.5 * semantic_similarity +
            0.2 * age_similarity
        )

        recommendations.append({
            'UserId': user['Id'],
            'UserName': user['Name'],
            'SimilarityScore': combined_similarity,
            'Distance': distance
        })

    recommendations = sorted(recommendations, key=lambda x: x['SimilarityScore'], reverse=True)[:top_n]
    return jsonify(recommendations)


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


if __name__ == '__main__':
    app.run(debug=True)
