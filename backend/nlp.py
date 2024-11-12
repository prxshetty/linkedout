import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List
import re

nltk.download('punkt', quiet = False)
nltk.download('stopwords', quiet = False)
nltk.download('wordnet', quiet = False)

def download_nltk_data():
    required_packages = ['punkt', 'stopwords', 'wordnet']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)

download_nltk_data()
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def extract_keywords(text: str) -> List[str]:
    print(len(text))
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    try:
        tokens = word_tokenize(text.lower())
    except LookupError:
        tokens = text.lower().split()
    keywords = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words and len(token) > 2]
    return list(dict.fromkeys(keywords))

def update_resume(resume_text: str, keywords: list = []) -> str:
    
    return updated_resume