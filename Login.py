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


def Login():
    # Title
    st.title("_TrendBlaze: Ignite Your TikTok Plays_ :bar_chart:")

    # Header
    st.write("To get started, please enter your TikTok username to see your personalized data, track your video performance and audience engagement. Dive into your stats and gain valuable insights to better understand your TikTok presence.")

    # Example
    st.info("Example usernasme: thehutchinsons")
    # Input username
    user_name = st.text_input('Input Your Tiktok Username')

    # Set comment count to default 0 to avoid error
    comment_count = 0

    # Initializing values
    followers,likes,videos,followering,diggCount = 0,0,0,0,0

    # Button to trigger analytics
    if st.button('Get your analytics'):
        asyncio.run(get_users(user_name))
        comment_count = asyncio.run(get_comment_count(user_name))

    # Load and display tabular data if username is provided
    if user_name:
        st.write(f"Analyzing data for: {user_name}")
        
        # Load in existing data to test
        df = pd.read_csv('userStats.csv')
        
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        
        followers = df['followerCount'].sum()
        likes = df['heartCount'].sum()
        videos = df['videoCount'].sum()
        following = df['followingCount'].sum()
        diggCount = df['diggCount'].sum()
        
        col1.metric("Total Followers", f"{followers:,}")
        col2.metric("Total Likes", f"{likes:,}")
        col3.metric("Total Videos", f"{videos:,}")
        col4.metric("Total Following", f"{following:,}")
        col5.metric("Total Comment Count", f"{comment_count:,}")
        col6.metric("Total Digg Count (Likes given to other users)", f"{diggCount:,}")

        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(name='Metrics', x=['Followers', 'Likes', 'Videos', 'Following', 'Comments',"Digg Count"],
                y=[followers, likes, videos, following, comment_count,diggCount])
        ])

        # Update layout for better readability
        fig.update_layout(
            title='TikTok Analytics Overview',
            xaxis_title='Metrics',
            yaxis_title='Count',
            yaxis_type='log',  # Using log scale due to large differences in values
            height=600
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    return followers,likes,videos,followering,diggCount