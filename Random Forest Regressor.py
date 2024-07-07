import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load your dataset
df = pd.read_excel('dataset-tiktok.xlsx')
df = pd.DataFrame(df)

# DateTime features
df['createTimeISO'] = pd.to_datetime(df['createTimeISO'], errors='coerce')
df['hour'] = df['createTimeISO'].dt.hour
df['day_of_week'] = df['createTimeISO'].dt.dayofweek
df['day'] = df['createTimeISO'].dt.day
df['month'] = df['createTimeISO'].dt.month

# Make verification and download binary
df['authorMeta/verified'] = df['authorMeta/verified'].astype(int)
df['downloaded'] = df['downloaded'].astype(int)

# Get text length
df['text_length'] = df['text'].str.len()

# Fill missing values
df.loc[:, df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna('NA')
df.loc[:, df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).fillna(0)

# Function to convert 'B' and 'M' to numerical values
def convert_to_numeric(value):
    if pd.isna(value):
        return 0
    elif 'B' in value:
        return float(value.replace('B', '')) * 1e9
    elif 'M' in value:
        return float(value.replace('M', '')) * 1e6
    else:
        return float(value)

# Apply the function to the 'searchHashtag/views' column
df['searchHashtag/views'] = df['searchHashtag/views'].apply(convert_to_numeric)

# Sentiment analysis
sid = SentimentIntensityAnalyzer()
df['sentiment'] = df['text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

# List of text columns to process separately
text_columns = ['effectStickers/0/name', 'hashtags/0/name', 'hashtags/0/title', 'hashtags/1/name',
                'hashtags/1/title', 'mentions/0', 'mentions/1', 'musicMeta/musicAuthor', 'musicMeta/musicName',
                'searchHashtag/name', 'text']

# Numerical features
numerical_features = ['authorMeta/fans', 'authorMeta/digg', 'authorMeta/heart', 'videoMeta/duration', 
                      'authorMeta/verified', 'downloaded', 'authorMeta/video', 'commentCount', 
                      'hour', 'day_of_week', 'sentiment', 'searchHashtag/views', 'text_length']

# Function to create features from text columns
def create_text_features(df, text_columns):
    sid = SentimentIntensityAnalyzer()
    text_features = pd.DataFrame(index=df.index)
    tfidf_vectorizers = {}
    tfidf_feature_mapping = {}  # To store the mapping
    start_index = 0
    
    for col in text_columns:
        # Length of text
        #text_features[f'{col}_length'] = df[col].str.len()
        
        # Sentiment analysis
        text_features[f'{col}_sentiment'] = df[col].apply(lambda x: sid.polarity_scores(str(x))['compound'])
        
        # TF-IDF features
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df[col])
        tfidf_vectorizers[col] = tfidf
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'{col}_tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
        
        # Store the mapping
        for i in range(tfidf_matrix.shape[1]):
            tfidf_feature_mapping[f'{col}_tfidf_{i}'] = col
        
        # Concatenate TF-IDF features
        text_features = pd.concat([text_features, tfidf_df], axis=1)
    
    return text_features, tfidf_vectorizers, tfidf_feature_mapping

# Create text features
text_features, tfidf_vectorizers, tfidf_feature_mapping = create_text_features(df, text_columns)

# Combine text features and numerical features
df_combined = pd.concat([df[numerical_features], text_features], axis=1)

# Target variable (assuming 'play_count' is the target)
y = df['playCount']
y_log = np.log1p(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df_combined, y_log, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor model
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

# Print feature importances
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': model_rf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)

# Map TF-IDF features to their text columns
def map_tfidf_features(feature_importances, tfidf_feature_mapping):
    feature_importances['Source Column'] = feature_importances['Feature'].map(tfidf_feature_mapping).fillna('Numerical')
    return feature_importances

# Map TF-IDF features to their text columns
feature_importances_mapped = map_tfidf_features(feature_importances, tfidf_feature_mapping)
print(feature_importances_mapped)

# Calculate and print accuracy metrics
from sklearn.metrics import mean_squared_error, r2_score

y_pred_train_log = model_rf.predict(X_train)
y_pred_test_log = model_rf.predict(X_test)

# Convert predictions back to original scale
y_pred_train = np.expm1(y_pred_train_log)
y_pred_test = np.expm1(y_pred_test_log)
y_train_orig = np.expm1(y_train)
y_test_orig = np.expm1(y_test)

train_mse = mean_squared_error(y_train_orig, y_pred_train)
test_mse = mean_squared_error(y_test_orig, y_pred_test)

train_r2 = r2_score(y_train_orig, y_pred_train)
test_r2 = r2_score(y_test_orig, y_pred_test)

print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')
print(f'Train R^2: {train_r2}')
print(f'Test R^2: {test_r2}')
