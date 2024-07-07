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
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from TagApi import Text_to_Hashtag

def data_input():
    # Title
    st.title("_TrendBlaze: Ignite Your TikTok Plays_ :bar_chart:")

    # Header
    st.write("Enter your TikTok post text to receive customised hashtag suggestions tailored to your content. Not sure what to type? Try out the sample text on the side to see how it works.")

    # Example text
    st.info("Example text input: Though we travel the world over to find the beautiful, we must carry it with us or we find it not. ")
    text = st.text_area("Enter video content description:")

    duration = st.number_input("Enter video duration (minutes):", min_value=0, step=1)

    hashtag_list = []  # Initialize hashtag_list outside the if statement

    if st.button('Get your hashtags'):
        if text and duration:
            hashtag_list = Text_to_Hashtag(text)
            st.write('Possible hashtags include: ')
            for hashtag in hashtag_list:
                st.write(f"- {hashtag}")
        else:
            st.error('Please fill in all fields before predicting.')

    return text, hashtag_list, duration