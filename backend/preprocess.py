import re
import string
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STEMMER = PorterStemmer()

def _load_stop_words():
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        return set(ENGLISH_STOP_WORDS)

_STOP_WORDS = _load_stop_words()

def preprocess_text(text):
    """Normalize, clean, remove stop words, and stem article text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [_STEMMER.stem(word) for word in text.split() if word not in _STOP_WORDS and len(word) > 2]
    return " ".join(tokens)
