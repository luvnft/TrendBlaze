from TikTokApi import TikTokApi
import asyncio
import os
import json
from helper import process_data
import pandas as pd

ms_token = os.environ.get("MS_TOKEN", "TjpJzt8U-oNPtyhl5iZHM6NGZweil0HghHbzRP8PMlQzORrlXCX_aPhjYctA8LhkY2cYzhmxV-MHyny_yYi7qcivLcuJAwCbHBGRk2EhUii2fRiLAV616-H5nWdfUJnJuQ-RwAJMzcJw")

async def get_hashtag(hashtag):
    async with TikTokApi() as api:
        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, headless=False)
        hashtag_data = await  api.hashtag(name=hashtag).info()
    challengeInfo = hashtag_data['challengeInfo']
    stats = challengeInfo['statsV2']
    return stats
