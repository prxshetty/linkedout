import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List
import re
import spacy
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from difflib import get_close_matches

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file")

openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key = OPENAI_API_KEY)
nltk.download('punkt', quiet = False)
nltk.download('stopwords', quiet = False)
nltk.download('wordnet', quiet = False)

data_science_keywords = {
    'machine learning', 'deep learning', 'neural networks', 'artificial intelligence', 'ai',
    'data analysis', 'data visualization', 'statistical analysis', 'big data',
    'python', 'r', 'sql', 'hadoop', 'spark', 'tensorflow', 'pytorch', 'keras',
    'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'tableau',
    'data mining', 'predictive modeling', 'natural language processing', 'nlp',
    'computer vision', 'time series analysis', 'a/b testing', 'etl',
    'data warehousing', 'cloud computing', 'aws', 'azure', 'gcp',
    'docker', 'kubernetes', 'version control', 'git', 'agile', 'scrum',
    'data structures', 'algorithms', 'software engineering', 'database management',
    'data ethics', 'data privacy', 'data governance', 'data pipeline',
    'feature engineering', 'model deployment', 'devops', 'mlops',
    'data engineering', 'data modeling', 'data governance', 'data quality',
    'data cleaning', 'data integration', 'data transformation', 'data extraction',
    'data modeling', 'data warehousing', 'data architecture', 'data security',
    'C++', 'C#', 'Java', 'JavaScript', 'TypeScript', 'PHP', 'Swift', 'Kotlin',
    'Go', 'Ruby', 'Rust', 'Scala', 'MATLAB', 'SAS', 'SPSS', 'Stata', 'Julia',
    'Fortran', 'COBOL', 'Ada', 'Erlang', 'Elixir', 'Haskell', 'OCaml', 'Racket',
    'Prolog', 'SQL', 'NoSQL', 'GraphQL', 'REST', 'SOAP', 'Web Services', 'Microservices',
    'data science', 'data analyst', 'data scientist', 'machine learning engineer',
    'deep learning engineer', 'ai engineer', 'data architect', 'data manager',
    'data governance', 'data quality', 'data integration', 'data transformation',
    'data extraction', 'data modeling', 'data warehousing', 'data architecture',
    'data security', 'data privacy', 'data ethics', 'data governance', 'data pipeline'
}

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
nlp = spacy.load('en_core_web_sm')

def extract_keywords(text: str) -> List[str]:
    print(f'Original text length: {len(text)}')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    try:
        tokens = word_tokenize(text.lower())
    except LookupError:
        tokens = text.lower().split()
    keywords = [
        lemmatizer.lemmatize(token)
        for token in tokens if token.isalpha() and token not in stop_words and len(token)>2
    ]
    unique_keywords = list(dict.fromkeys(keywords))
    technical_keywords = identify_technical_keywords(text, unique_keywords)
    return technical_keywords

def identify_technical_keywords(text:str, keywords:List[str]) -> List[str]:
    doc = nlp(text)
    ner_keywords = [ent.text.lower() for ent in doc.ents if ent.label_ in {
        # 'ORG',
        'PRODUCT', 
        'LANGUAGE', 
        'SKILL',
        'PERSON',
        # 'GPE',
        'WORK_OF_ART'
        }]
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 50, ngram_range = (1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    tfidf_keywords = [feature_names[i] for i in tfidf_scores.argsort()[-50:]]
    combine_keywords = set(ner_keywords + tfidf_keywords)

    technical_terms = []
    data_science_keywords_lower = {keyword.lower() for keyword in data_science_keywords}
    
    for word in combine_keywords:
        word_lower = word.lower()
        if word_lower in data_science_keywords_lower:
            original_case = next(k for k in data_science_keywords if k.lower() == word_lower)
            technical_terms.append(original_case)
        else:
            close_matches = get_close_matches(word_lower, data_science_keywords_lower, n=1, cutoff=0.35)
            if close_matches:
                original_case = next(k for k in data_science_keywords if k.lower() == close_matches[0])
                technical_terms.append(original_case)
            else:
                if any(token.pos_ in {'NOUN', 'PROPN'} for token in nlp(word)):
                    technical_terms.append(word)
    text_lower = text.lower()
    technical_terms.sort(key=lambda x: text_lower.count(x.lower()), reverse=True)
    seen = set()
    unique_terms = []
    for term in technical_terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique_terms.append(term)
    
    return unique_terms[:50]
    # for word in combine_keywords:
    #     if any(term in data_science_keywords for term in word.split()):
    #         technical_terms.append(word)
    #     else:
    #         if any (token.pos_ in {'NOUN', 'PROPN'} for token in nlp(word)):
    #             technical_terms.append(word)
    # text_lower = text.lower()
    # technical_terms.sort(key = lambda x: text_lower.count(x), reverse = True)
    # return list(dict.fromkeys(technical_terms))[:40]

def analyze_with_ai(text: str) -> List[str]:
    try:
        response = client.chat.completions.create(
            model = 'gpt-4o-mini',
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert technical recruiter specializing in data science and AI roles. 
                    Your task is to analyze job descriptions and extract key technical skills, tools, and requirements.
                    Focus on:
                    - Programming languages
                    - Data science libraries and frameworks
                    - Machine learning and deep learning tools
                    - Database technologies
                    - Data visualization tools
                    - Cloud platforms
                    - Version control systems
                    - Big data technologies
                    - Statistical analysis methods
                    
                    Return your response in this exact JSON format:
                    {"technical_skills": ["skill1", "skill2", "skill3", ...]}
                    
                    Limit your response to the most important 20 technical terms, prioritizing specific tools and technologies mentioned in the job description."""
                },
                {
                    "role": "user",
                    "content": f"Analyze this job description and extract key technical requirements:\n\n{text}"
                }
            ],
            temperature=0.2,
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        print("OpenAI Response:", response.choices[0].message.content)

        result = json.loads(response.choices[0].message.content)
        if isinstance(result, dict) and "technical_skills" in result:
            return result["technical_skills"]
        return []
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        return []
    

def update_resume(resume_text: str, keywords: list = []) -> str:
    
    return updated_resume