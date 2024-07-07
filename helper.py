def process_data(data):
    user_Info = data['userInfo']
    stats = user_Info['stats']
    return stats

def process_hashtag(data):
    challengeInfo = data['challengeInfo"']
    stats = challengeInfo['statsV2']
    return stats