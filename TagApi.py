from ritetag import RiteTagApi

access_token = "ab3028eceb77a074884906d136416b39999c3d87363b"
client = RiteTagApi(access_token)

def Text_to_Hashtag(text):
    stats = client.hashtag_suggestion_for_text([text])

    # Take the top3 hashtag exposure
    sorted_stats = sorted(stats, key=lambda x: x.exposure, reverse=True)
    top_3_hashtags = sorted_stats[:3]
    hashtag_list = []

    for data in top_3_hashtags:
        hashtag_list.append(data.hashtag)
    return hashtag_list
