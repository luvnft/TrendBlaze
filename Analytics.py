import streamlit as st
import pandas as pd
import numpy as np
import joblib
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from textblob import TextBlob

# Load your model and feature info
loaded_model = joblib.load('best_model.joblib')
feature_names, feature_means = joblib.load('feature_info.joblib')
sloaded_model = joblib.load('sbest_model.joblib')
sfeature_names, sfeature_means = joblib.load('sfeature_info.joblib')

def analytics(text, hashtags, duration, comment_count, hashtag_views, follower_count, heart_count, digg_count, video_count):
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


    def process_input(text, hashtags, duration, comment_count, hashtag_views, follower_count, heart_count, digg_count, video_count):
        # TF-IDF vectorization
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
        user_tfidf = tfidf_matrix[-1]

        # One-hot encoding for hashtags
        all_hashtags = list(set(hashtags))
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        one_hot_encoder.fit([[tag] for tag in all_hashtags])
        user_hashtags_encoded = one_hot_encoder.transform([[tag] for tag in hashtags])

        # Convert sparse matrices to dense for concatenation
        user_tfidf_dense = user_tfidf.toarray()
        user_hashtags_dense = user_hashtags_encoded.toarray().sum(axis=0)

        # Create a DataFrame for the user data
        user_df = pd.DataFrame({
            'commentCount': [comment_count],
            'searchHashtag/views': [hashtag_views],
            'authorMeta/fans': [follower_count],
            'authorMeta/heart': [heart_count],
            'videoMeta/duration': [duration],
            'authorMeta/digg': [digg_count],
            'text_length': [len(text)],
            'authorMeta/video': [video_count]
        })

        # Concatenate all features
        user_features = np.hstack((user_df.values, user_tfidf_dense, user_hashtags_dense.reshape(1, -1)))
        
        return user_features


    def process_smaller_input(text, hashtags, duration, hashtag_views):
        # TF-IDF vectorization
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
        user_tfidf = tfidf_matrix[-1]

        # One-hot encoding for hashtags
        all_hashtags = list(set(hashtags))
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        one_hot_encoder.fit([[tag] for tag in all_hashtags])
        user_hashtags_encoded = one_hot_encoder.transform([[tag] for tag in hashtags])

        # Convert sparse matrices to dense for concatenation
        user_tfidf_dense = user_tfidf.toarray()
        user_hashtags_dense = user_hashtags_encoded.toarray().sum(axis=0)

        # Create a DataFrame for the user data
        user_df = pd.DataFrame({
            'searchHashtag/views': [hashtag_views],
            'videoMeta/duration': [duration],
            'text_length': [len(text)]
        })

        # Concatenate all features
        smaller_user_features = np.hstack((user_df.values, user_tfidf_dense, user_hashtags_dense.reshape(1, -1)))
        
        return smaller_user_features

    def analyze_text(text):
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        suggestions = []
        
        # Text length suggestions
        if len(text) < 50:
            suggestions.append(("length", "Increase text length to 50 characters"))
        elif len(text) > 300:
            suggestions.append(("length", "Decrease text length to 300 characters"))
        
        # Sentiment suggestions
        if sentiment < -0.1:
            suggestions.append(("sentiment", "Use more positive language"))
        elif sentiment > 0.5:
            suggestions.append(("sentiment", "Maintain very positive tone"))
        
        # Content suggestions
        if '?' not in text:
            suggestions.append(("question", "Add a question"))
        if '!' not in text:
            suggestions.append(("exclamation", "Add an exclamation mark"))
        
        return suggestions

    def apply_suggestions(text, suggestions):
        new_text = text
        new_hashtags = hashtags.copy()
        
        for suggestion in suggestions:
            if suggestion[0] == "length":
                if "Increase" in suggestion[1]:
                    new_text = new_text.ljust(50, ' ')
                elif "Decrease" in suggestion[1]:
                    new_text = new_text[:300]
            elif suggestion[0] == "sentiment":
                # For simplicity, we'll just append a positive phrase
                new_text += " This is amazing!"
            elif suggestion[0] == "hashtags":
                if "Increase" in suggestion[1]:
                    new_hashtags.extend(["trending", "fyp"])
                elif "Decrease" in suggestion[1]:
                    new_hashtags = new_hashtags[:5]
            elif suggestion[0] == "question":
                new_text += " What do you think?"
            elif suggestion[0] == "exclamation":
                new_text += "!"
        
        return new_text, new_hashtags

    def find_best_combination(text, hashtags, duration, suggestions):
        best_score = 0
        best_combination = []
        original_features = process_smaller_input(text, hashtags, duration,hashtag_views)
        original_prediction = sloaded_model.predict(prepare_prediction_data(original_features, sfeature_names, sfeature_means))[0]
        
        for r in range(1, len(suggestions) + 1):
            for combination in itertools.combinations(suggestions, r):
                new_text, new_hashtags = apply_suggestions(text, hashtags, combination)
                new_features = process_smaller_input(new_text, new_hashtags, duration,hashtag_views)
                new_prediction = sloaded_model.predict(prepare_prediction_data(new_features, sfeature_names, sfeature_means))[0]
                
                if new_prediction > best_score:
                    best_score = new_prediction
                    best_combination = combination
        
        increase = best_score - original_prediction
        return best_combination, increase

    st.title('TikTok Video Predictor and Optimizer')

    if st.button('Predict and Optimize'):

        user_features = process_input(text, hashtags, duration, comment_count, hashtag_views, follower_count, heart_count, digg_count, video_count)
        
        # Prepare the data
        prepared_data = prepare_prediction_data(user_features, feature_names, feature_means)
        
        # Make prediction
        prediction = loaded_model.predict(prepared_data)
        
        st.success(f'The predicted play count is: {prediction[0]}')
        
        # Get suggestions
        suggestions = analyze_text(text, hashtags)
        
        if suggestions:
            st.subheader("Optimization Results:")
            best_combination, increase = find_best_combination(text, hashtags, duration, suggestions)
            
            st.write(f"Potential increase in views: {increase:.2f}")
            st.write("Best combination of suggestions:")
            for suggestion in best_combination:
                st.write(f"- {suggestion[1]}")
            
            # Apply best suggestions
            optimized_text, optimized_hashtags = apply_suggestions(text, hashtags, best_combination)
            st.subheader("Optimized Content:")
            st.write(f"Text: {optimized_text}")
            st.write(f"Hashtags: {', '.join(optimized_hashtags)}")
        else:
            st.write("Your content looks optimal! No specific suggestions at this time.")
