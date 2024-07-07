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



def Hashtag(hashtag_list):
    # Title
    st.title("_TrendBlaze: Ignite Your TikTok Plays_ :bar_chart:")

    # Header
    st.write("Want to see how your hashtags are performing? Click the button to get detailed analytics and insights. Understand the reach, engagement, and effectiveness of each hashtag to optimize your TikTok strategy.")

    st.write("Added Hashtags:")

    for hashtag in hashtag_list:
        st.write(f"- {hashtag}")

    total_view_count = 0

    if hashtag_list:
        if st.button('Get your analytics'):
            stats_graph = {'Hashtag': [], 'Video Count': [], 'View Count': [], 'Popularity': []}
            for hashtag in hashtag_list:
                st.subheader(f"Analytics for Hashtag: {hashtag}")
                stats = asyncio.run(get_hashtag(hashtag))
                col1, col2, col3 = st.columns(3)
                video_count = int(stats['videoCount'])
                view_count = int(stats['viewCount'])
                popularity = view_count / video_count if video_count > 0 else 0

                col1.metric("Total videos", video_count)
                col2.metric("Total views", view_count)
                col3.metric("Popularity (Avg views per video)", f"{popularity:.2f}")
                
                stats_graph['Hashtag'].append(hashtag)
                stats_graph['Video Count'].append(video_count)
                stats_graph['View Count'].append(view_count)
                stats_graph['Popularity'].append(popularity)

                total_view_count += view_count

            # Create DataFrame
            df = pd.DataFrame(stats_graph)

            # Set a color palette
            num_colors = len(df['Hashtag'])
            colors = sns.color_palette("husl", num_colors)

            # Create three separate bar charts
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
            
            # Video Count Chart
            bars1 = ax1.bar(df['Hashtag'], df['Video Count'], color=colors)
            ax1.set_ylabel('Video Count')
            ax1.set_title('Video Count Comparison')
            ax1.tick_params(axis='x', rotation=45)

            # View Count Chart
            bars2 = ax2.bar(df['Hashtag'], df['View Count'], color=colors)
            ax2.set_ylabel('View Count')
            ax2.set_title('View Count Comparison')
            ax2.tick_params(axis='x', rotation=45)

            # Popularity Chart
            bars3 = ax3.bar(df['Hashtag'], df['Popularity'], color=colors)
            ax3.set_ylabel('Popularity (Avg Views per Video)')
            ax3.set_title('Popularity Comparison')
            ax3.tick_params(axis='x', rotation=45)

            # Add value labels on top of each bar
            def add_value_labels(ax, bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom')

            add_value_labels(ax1, bars1)
            add_value_labels(ax2, bars2)
            add_value_labels(ax3, bars3)

            plt.tight_layout()

            # Display the plots in Streamlit
            st.pyplot(fig)

            # Display the data table
            st.write("Data Table:")
            st.dataframe(df)

            # Calculate and display percentages
            total_views = df['View Count'].sum()
            total_videos = df['Video Count'].sum()
            
            df['View Percentage'] = (df['View Count'] / total_views * 100).round(2)
            df['Video Percentage'] = (df['Video Count'] / total_videos * 100).round(2)
            
            st.write("Percentages and Popularity:")
            st.dataframe(df[['Hashtag', 'View Count', 'View Percentage', 'Video Count', 'Video Percentage', 'Popularity']])
    
    return total_view_count