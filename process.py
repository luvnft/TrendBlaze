import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Load your model and feature info
loaded_model = joblib.load('best_model.joblib')
feature_names, feature_means = joblib.load('feature_info.joblib')

def prepare_prediction_data(prediction_data, feature_names, feature_means):
    # Convert to numpy array if it's not already
    if isinstance(prediction_data, pd.DataFrame):
        prediction_data = prediction_data.to_numpy()
    elif not isinstance(prediction_data, np.ndarray):
        prediction_data = np.array(prediction_data)
    
    # Ensure prediction_data is 2D
    if prediction_data.ndim == 1:
        prediction_data = prediction_data.reshape(1, -1)
    
    # Create a DataFrame with all features, filling missing ones with means
    full_data = pd.DataFrame(columns=feature_names)
    for i, feature in enumerate(feature_names):
        if i < prediction_data.shape[1]:
            full_data[feature] = prediction_data[:, i]
        else:
            full_data[feature] = feature_means[feature]
    
    # Fill any NaN values with means
    full_data = full_data.fillna(feature_means)
    
    return full_data.to_numpy()

def process_input(followers, likes, videos, diggCount, comment_count, duration, hashtag_views, text, hashtags):
    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    user_tfidf = tfidf_matrix[-1]  # Corrected this line

    # One-hot encoding for hashtags
    all_hashtags = list(set(hashtags))
    if all_hashtags:  # Only perform one-hot encoding if there are hashtags
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        one_hot_encoder.fit([[tag] for tag in all_hashtags])
        user_hashtags_encoded = one_hot_encoder.transform([[tag] for tag in hashtags])
        user_hashtags_dense = user_hashtags_encoded.toarray().sum(axis=0)
    else:
        # If there are no hashtags, create an empty array
        user_hashtags_dense = np.array([])

    # Convert sparse matrices to dense for concatenation
    user_tfidf_dense = user_tfidf.toarray()

    # Create a DataFrame for the user data
    user_df = pd.DataFrame({
        'commentCount': [comment_count],
        'searchHashtag/views': [hashtag_views],
        'authorMeta/fans': [followers],
        'authorMeta/heart': [likes],
        'videoMeta/duration': [duration],
        'authorMeta/digg': [diggCount],
        'text_length': [len(text)],
        'authorMeta/video': [videos]
    })

    # Concatenate all features
    user_features = np.hstack((user_df.values, user_tfidf_dense, user_hashtags_dense.reshape(1, -1)))
    
    return user_features

def analytics(followers, likes, videos, diggCount, comment_count, duration, hashtag_views, text, hashtags):
    st.title("_TrendBlaze: Ignite Your TikTok Plays_ :bar_chart:")

    # Header
    st.write("Compare the modified playcount with our auto-generated hashtags to the original playcount to see how powerful our web app is!")

    if st.button('Final Analytics'):
        st.write("Added Hashtags:")
        for hashtag in hashtags:
            st.info(hashtag)

        # Process input with original data
        user_features_modified = process_input(followers, likes, videos, diggCount, comment_count, duration, hashtag_views, text, hashtags)
        
        # Process input with empty hashtag list and zero views
        user_features_original = process_input(followers, likes, videos, diggCount, comment_count, duration, 0, text, [])
        
        # Prepare the data
        prepared_data_modified = prepare_prediction_data(user_features_modified, feature_names, feature_means)
        prepared_data_original = prepare_prediction_data(user_features_original, feature_names, feature_means)
        
        # Make predictions
        prediction_original = loaded_model.predict(prepared_data_original)
        prediction_modified = loaded_model.predict(prepared_data_modified)
        
        # Display predictions
        st.success(f'The prediction playcount with original data: {prediction_original[0]}')
        st.success(f'The prediction playcount with our suggested hashtags: {prediction_modified[0]}')
        
        # Create comparison graph
        fig, ax = plt.subplots()
        predictions = [prediction_original[0], prediction_modified[0]]
        labels = ['Original', 'Modified']
        ax.bar(labels, predictions)
        ax.set_ylabel('Predicted Playcount')
        ax.set_title('Comparison of Predictions')
        
        # Display the graph
        st.pyplot(fig)
