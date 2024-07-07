# Red_Romance
For TikTok Techjam

## TrendBlaze: TikTok Analytics and Optimization Tool

## Description

TrendBlaze is a comprehensive TikTok play count predictor and optimizer web application, designed to help content creators maximize their engagement and reach on TikTok. This innovative tool leverages advanced machine learning algorithms to predict the play count of TikTok videos based on various features and provides suggestions for optimizing these predictions. By using TrendBlaze, creators can gain valuable insights and make data-driven decisions to enhance their content performance.


## Features

1. **Introduction**: provides an overview of the TrendBlaze algorithm, explaining how the predictor and optimizer work. It details how users can utilize the tool to forecast play counts and enhance their content's performance. The introduction also highlights up to ten important features that influence the play count of TikTok videos, such as comment count, hashtag views, follower count, and etc.
2. **Login**: users can enter their TikTok username to access their personal analytics. By clicking ‘Get your analytics’, TrendBlaze will display the total followers, likes, videos, comments, and other relevant statistics to the user, as well as a bar chart summarizing these metrics, providing users with a visual representation of their profile's performance.
3. **Data Input**: allows users to enter the captions for their videos and specify the video duration. Based on this information, the app recommends suitable hashtags to include in their posts.
4. **Hashtags**: offers detailed analytics for each recommended hashtag. Users can view data on the number of videos, views, and overall popularity associated with each hashtag. A bar chart compares these categories for the recommended hashtags, helping users make informed decisions about which hashtags to use.
5. **Analytics**: displays the predicted play count for a video before and after adding the recommended hashtags. It shows the impact of the suggested optimizations, with a bar chart illustrating the difference in predicted play counts. This allows users to see the potential benefits of implementing the recommendations provided by TrendBlaze.

## Installation

1. Clone the repository:
```
git clone https://github.com/manyuhaochi214/TrendBlaze.git
cd trendblaze
```
3. Install the required packages:
```
pip install -r requirements.txt
```
3. Set up your TikTok API credentials:
- Obtain a MS_TOKEN from TikTok
- Set the MS_TOKEN as an environment variable or update it in the relevant Python files

## Usage

Run the Streamlit app:
```
streamlit run app.py
```
Follow the steps in the sidebar to navigate through the application:
1. Introduction
2. Login
3. Data Input
4. Hashtag Analysis
5. Analytics

## File Structure

- `app.py`: Main application file that runs the Streamlit interface
- `Introduction.py`: Contains the introduction page content
- `Login.py`: Handles user login and fetches account statistics
- `Hashtag.py`: Manages hashtag analysis and suggestions
- `dataInput.py`: Handles user input for video content and details
- `process.py`: Contains the core logic for predictive analytics and optimization
- `getUser.py`: Fetches user data from TikTok API
- `getHashTag.py`: Retrieves hashtag data from TikTok API
- `getTrending.py`: Fetches trending videos from TikTok
- `helper.py`: Contains utility functions for data processing
- `Analytics.py`: Provides detailed analytics and optimization suggestions
- `Random Forest Regressor.py`: Implements the Random Forest model for predictions
- `TextToHashtag.py`: Converts text content to relevant hashtags
- `TagApi.py`: Interfaces with the RiteTag API for hashtag suggestions

## Models and Data

- `best_model.joblib`: Trained machine learning model for play count prediction
- `feature_info.joblib`: Feature information for the main prediction model
- `sbest_model.joblib`: Smaller trained model for quick predictions
- `sfeature_info.joblib`: Feature information for the smaller model

## Dependencies

Main dependencies include:
- streamlit
- pandas
- numpy
- scikit-learn
- TikTokApi
- plotly
- matplotlib
- seaborn
- joblib
- xgboost
- ritetag


## Notes

- This application uses the Unofficial TikTok API and Reitag API, which both acknowledge the software can be used without restrictions.
- The predictive models are based on historical data and may need periodic retraining for optimal performance.
- You may refer to the video here for the overview of the TrendBlaze Webapp, [click here](https://youtu.be/QXlmyMcSFJc)

## Contributing

Contributions to TrendBlaze are welcome! Please feel free to submit pull requests, create issues or spread the word.

## License

this project is none licenced.
