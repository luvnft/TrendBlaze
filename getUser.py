from TikTokApi import TikTokApi
import asyncio
import os
import json
from helper import process_data
import pandas as pd

ms_token = os.environ.get("MS_TOKEN", "TjpJzt8U-oNPtyhl5iZHM6NGZweil0HghHbzRP8PMlQzORrlXCX_aPhjYctA8LhkY2cYzhmxV-MHyny_yYi7qcivLcuJAwCbHBGRk2EhUii2fRiLAV616-H5nWdfUJnJuQ-RwAJMzcJw")

async def get_users(name):
    async with TikTokApi() as api:
        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, headless=False)
        user_data = await api.user(username=name).info()
    # process json file into a dict
    user_stats = process_data(user_data)
    stats_list = []
    stats_list.append(user_stats)
    # convert data to a dataframe
    df = pd.DataFrame.from_dict(stats_list)
    # convert to csv file
    df.to_csv('userStats.csv', index=False)

async def get_comment_count(name):
    async with TikTokApi() as api:
        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, headless=False)
        comment_Count = 0
        async for video in api.user(username=name).videos():
            count = video.stats['commentCount']
            comment_Count += count
    return comment_Count

