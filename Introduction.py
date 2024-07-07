import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from getUser import get_users
from getUser import get_comment_count
import asyncio
from getHashTag import get_hashtag
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def Intro():
    data = {
    'Feature': ['commentCount', 'searchHashtag/views', 'searchHashtag/name_tfidf_0', 'authorMeta/fans',
                'authorMeta/heart', 'text_tfidf_75', 'videoMeta/duration', 'authorMeta/digg',
                'text_length', 'authorMeta/video'],
    'Importance': [0.490836, 0.249197, 0.063102, 0.031031, 0.027671, 0.013797, 0.013658, 0.007347,
                0.007225, 0.005188],
    'Source Column': ['Numerical', 'Numerical', 'searchHashtag/name', 'Numerical', 'Numerical',
                    'text', 'Numerical', 'Numerical', 'Numerical', 'Numerical']
    }

    st.title("_TrendBlaze: Ignite Your TikTok Plays_ :bar_chart:")

    st.write("""

    Welcome to TrendBlaze, a powerful tool designed to maximize your TikTok post's visibility and engagement. Utilize the latest AI technology to discover and generate the best hashtags tailored for your content, ensuring your posts reach a wider audience and achieve higher interaction rates. Start optimizing your hashtags today and watch your TikTok presence soar!
    """)

    st.info("""
    **Did you know?** 
    While factors like follower count and comment engagement play a role in TikTok's algorithm, our research shows that hashtags are a key variable you can control to increase your reach. The right combination of hashtags can expose your content to a broader, more relevant audience.
    """)

    st.subheader("How It Works")
    st.write("""
    1. Input your username
    2. Input post content
    3. Get personalized hashtag recommendations
    4. Apply to your TikTok content and watch your playcounts soar!
    """)

    df = pd.DataFrame(data)

    # Sort DataFrame by Importance in descending order
    df = df.sort_values('Importance', ascending=False)

    # Create bar chart using Plotly
    fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                color='Source Column', 
                title='Feature Importance for TikTok Analytics',
                labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
                height=600)

    # Customize layout
    fig.update_layout(yaxis={'categoryorder':'total ascending'})

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Display the data table
    st.subheader("Feature Importance Data")
    st.dataframe(df)