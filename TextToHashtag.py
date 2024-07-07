import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def text_to_hashtag(text):
    # Use NLTK to extract keywords from content
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    keywords = [word for (word, tag) in tags if tag.startswith('NN') or tag == 'NNP']
    return keywords

print(text_to_hashtag('The tardigrade, also known as the water bear, can survive in extreme conditions that would be fatal to most other life forms. These microscopic animals can withstand temperatures from near absolute zero to well above the boiling point of water, pressure six times greater than that found in the deepest ocean trenches, and even the vacuum of space.'))