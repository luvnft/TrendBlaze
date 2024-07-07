import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
import xgboost as xgb
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Train the model with feature selected data
df = pd.read_excel('dataset-tiktok.xlsx')
df = pd.DataFrame(df)
datas = df[['commentCount','searchHashtag/views','searchHashtag/name','authorMeta/fans','authorMeta/heart',
           'text','videoMeta/duration','authorMeta/digg','text_length','authorMeta/video','playCount']]
datas[0:5]

# Handle categorical data
ohe = OneHotEncoder()
searchHashtag_name_encoded = ohe.fit_transform(datas[['searchHashtag/name']]).toarray()

# Handle text data
tfidf = TfidfVectorizer()
text_tfidf = tfidf.fit_transform(datas['text']).toarray()

# Combine the processed features
features = pd.concat([datas.drop(columns=['text', 'searchHashtag/name', 'playCount']), 
                      pd.DataFrame(searchHashtag_name_encoded), 
                      pd.DataFrame(text_tfidf)], axis=1)

# Convert all column names to strings
features.columns = features.columns.astype(str)

# Normalize numerical features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Define target variable
target = datas['playCount']

smaller_features = features.drop(columns=['commentCount','authorMeta/fans','authorMeta/heart','authorMeta/digg','authorMeta/video'], axis=1)
smaller_features.columns = smaller_features.columns.astype(str)
sfeatures_scaled = scaler.fit_transform(smaller_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

# Make predictions with the XGBoost model
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"XGBoost Mean Squared Error: {mse_xgb}")
print(mse_svr/mse_xgb)


# Define parameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Initialize GridSearchCV for XGBoost
grid_search_xgb = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), param_grid_xgb, refit=True, verbose=2, cv=3)
grid_search_xgb.fit(X_train, y_train)

# Best parameters for XGBoost
print("Best parameters for XGBoost:", grid_search_xgb.best_params_)

# Evaluate the best XGBoost model
y_pred_xgb_tuned = grid_search_xgb.predict(X_test)
mse_xgb_tuned = mean_squared_error(y_test, y_pred_xgb_tuned)
print(f"Tuned XGBoost Mean Squared Error: {mse_xgb_tuned}")

best_model = grid_search_xgb.best_estimator_

# Split data into training and testing sets
sX_train, sX_test, sy_train, sy_test = train_test_split(sfeatures_scaled, target, test_size=0.2, random_state=42)

param_grid_xgb = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Initialize GridSearchCV for XGBoost
sgrid_search_xgb = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), param_grid_xgb, refit=True, verbose=2, cv=3)
sgrid_search_xgb.fit(sX_train, sy_train)

# Best parameters for XGBoost
print("Best parameters for XGBoost:", sgrid_search_xgb.best_params_)

# Evaluate the best XGBoost model
sy_pred_xgb_tuned = sgrid_search_xgb.predict(sX_test)
smse_xgb_tuned = mean_squared_error(sy_test, sy_pred_xgb_tuned)
sbest_model = sgrid_search_xgb.best_estimator_


# Assuming X_train is your training data as a numpy array
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)

# If you don't have feature names, we'll create generic ones
feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
feature_means = dict(zip(feature_names, imputer.statistics_))

# Save these for later use
joblib.dump((feature_names, feature_means), 'feature_info.joblib')

# Assuming X_train is your training data as a numpy array
simputer = SimpleImputer(strategy='mean')
simputer.fit(sX_train)

# If you don't have feature names, we'll create generic ones
sfeature_names = [f'feature_{i}' for i in range(sX_train.shape[1])]
sfeature_means = dict(zip(feature_names, simputer.statistics_))

# Save these for later use
joblib.dump((sfeature_names, sfeature_means), 'sfeature_info.joblib')