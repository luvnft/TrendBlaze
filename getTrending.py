from TikTokApi import TikTokApi
import asyncio
import os
import json

ms_token = os.environ.get("TjpJzt8U-oNPtyhl5iZHM6NGZweil0HghHbzRP8PMlQzORrlXCX_aPhjYctA8LhkY2cYzhmxV-MHyny_yYi7qcivLcuJAwCbHBGRk2EhUii2fRiLAV616-H5nWdfUJnJuQ-RwAJMzcJw", None)

async def trending_videos():
    videos_data = []
    async with TikTokApi() as api:
        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, headless=False)
        async for video in api.trending.videos(count=30):
            video_dict = video.as_dict
            videos_data.append(video_dict)
    
    return videos_data

async def main():
    videos = await trending_videos()
    
    # Export to JSON
    with open('trending_videos.json', 'w', encoding='utf-8') as f:
        json.dump(videos, f, ensure_ascii=False, indent=4)
    
    print(f"Exported {len(videos)} videos to trending_videos.json")

if __name__ == "__main__":
    asyncio.run(main())