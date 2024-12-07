# modules/feedback.py

def get_feedback_subcategories(feedbacks, user_subcategories_df, positive=True):
    rating_value = 5 if positive else 1
    relevant_feedback = feedbacks[feedbacks['Rating'] == rating_value]
    feedback_subcategories = set()
    for _, feedback in relevant_feedback.iterrows():
        target_user_subcategories = user_subcategories_df[
            user_subcategories_df['UserId'] == feedback['TargetUserId']
        ]['SubcategoryId']
        feedback_subcategories.update(target_user_subcategories)
    return feedback_subcategories

def adjust_similarity_with_feedback(similarity, user_subcategories, positive_subs, negative_subs):
    adjustment = 0
    for sub_id in user_subcategories:
        if sub_id in positive_subs:
            adjustment += 0.05
        if sub_id in negative_subs:
            adjustment -= 0.05
    return similarity + adjustment

def adjust_profile_with_feedback(user_profile, user_subcats, positive_subs, negative_subs, subcategories_df):
    positive_terms = [
        subcategories_df[subcategories_df['Id'] == sub]['Name'].values[0]
        for sub in positive_subs if sub in user_subcats
    ]
    negative_terms = [
        subcategories_df[subcategories_df['Id'] == sub]['Name'].values[0]
        for sub in negative_subs if sub in user_subcats
    ]
    adjusted_profile = user_profile + ' ' + ' '.join(positive_terms)
    for term in negative_terms:
        adjusted_profile = adjusted_profile.replace(term, '')
    return adjusted_profile.strip()
