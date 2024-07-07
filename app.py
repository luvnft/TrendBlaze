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
from Introduction import Intro
from Login import Login
from Hashtag import Hashtag
from dataInput import data_input
from process import analytics

def main():
    # Default settings
    if 'followers' not in st.session_state:
        st.session_state.followers = 0
    if 'likes' not in st.session_state:
        st.session_state.likes = 0
    if 'videos' not in st.session_state:
        st.session_state.videos = 0
    if 'following' not in st.session_state:
        st.session_state.following = 0
    if 'diggCount' not in st.session_state:
        st.session_state.diggCount = 0
    if 'comment_count' not in st.session_state:
        st.session_state.comment_count = 0
    if 'hashtag_list' not in st.session_state:
        st.session_state.hashtag_list = []
    if 'duration' not in st.session_state:
        st.session_state.duration = 0
    if 'total_view_count' not in st.session_state:
        st.session_state.total_view_count = 0
    if 'text' not in st.session_state:
        st.session_state.text = ''


    # Home Page
    st.set_page_config(layout="wide")

    # Sidebar content
    st.sidebar.image("https://freepnglogo.com/images/all_img/1691751088logo-tiktok-png.png", width=250)
    st.sidebar.title("TrendBlaze")
    choice = st.sidebar.radio("Follow the steps", ["Introduction","Login","Data Input", "Hashtag","Analytics"])

    st.sidebar.title("Optimization Steps")
    st.sidebar.write("Follow these steps to optimize your hashtags:")
    steps = [
        "Enter post details",
        "Choose content category",
        "Review AI suggestions",
        "Copy optimized hashtags"
        ]
    for i, step in enumerate(steps, 1):
        st.sidebar.write(f"{i}. {step}")

    # Navigation
    if choice=="Introduction":
        Intro()

    elif choice=="Login":
        st.session_state.followers, st.session_state.likes,st.session_state.videos, st.session_state.following,st.session_state.diggCount = Login()

    elif choice=="Data Input":
        st.session_state.text,st.session_state.hashtag_list, st.session_state.duration= data_input()  # This will now always return a list, even if empty

    elif choice=="Hashtag":
        st.session_state.total_view_count = Hashtag(st.session_state.hashtag_list)

    elif choice=="Analytics":
        analytics(st.session_state.followers,st.session_state.likes,st.session_state.videos,st.session_state.diggCount,st.session_state.comment_count,st.session_state.duration,st.session_state.total_view_count,st.session_state.text,st.session_state.hashtag_list)

if __name__=="__main__":
    main()
