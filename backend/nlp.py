import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict
import re
import spacy
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import os
from dotenv import load_dotenv
from difflib import get_close_matches

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file")

# openai.api_key = OPENAI_API_KEY
# client = OpenAI(api_key = OPENAI_API_KEY)
# nltk.download('punkt', quiet = False)
# nltk.download('stopwords', quiet = False)
# nltk.download('wordnet', quiet = False)

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
def escape_latex(text: str) -> str:
    """
    Escapes LaTeX special characters in text to prevent compilation errors.
    
    Args:
        text (str): The text to escape.
    
    Returns:
        str: The escaped text.
    """
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def validate_latex(latex_code: str) -> bool:
    return latex_code.strip().startswith(r'\documentclass') and latex_code.strip().endswith(r'\end{document}')

def extract_sections(latex_code: str) -> Dict[str, List[str]]:
    """Extract sections and their bullet points from LaTeX code."""
    sections = {
        'experience': [],
        'projects': [],
        'technical_skills': []
    }
    exp_pattern = r'\\section{Experience}(.*?)(?=\\section|\\end{document})'
    exp_match = re.search(exp_pattern, latex_code, re.DOTALL)
    if exp_match:
        sections['experience'] = re.findall(r'\\resumeItem{(.*?)}', exp_match.group(1))
    
    proj_pattern = r'\\section{Projects}(.*?)(?=\\section|\\end{document})'
    proj_match = re.search(proj_pattern, latex_code, re.DOTALL)
    if proj_match:
        sections['projects'] = re.findall(r'\\resumeItem{(.*?)}', proj_match.group(1))
    
    tech_pattern = r'\\section{Technical Skills}(.*?)(?=\\section|\\end{document})'
    tech_match = re.search(tech_pattern, latex_code, re.DOTALL)
    if tech_match:
        sections['technical_skills'] = [tech_match.group(1)]
    
    return sections

def replace_sections(latex_code: str, optimized_sections: Dict[str, List[str]]) -> str:
    result = latex_code
    
    for section_key, items in optimized_sections.items():
        if not items:
            continue
        
        if section_key in ['experience', 'projects']:
            section_pattern = rf'(\\section{{{section_key.capitalize()}}}.*?\\resumeSubHeadingListStart)(.*?)(\\resumeSubHeadingListEnd)'
            section_match = re.search(section_pattern, result, re.DOTALL | re.IGNORECASE)
            
            if section_match:
                formatted_items = []
                for item in items:
                    cleaned_item = (
                        item.strip()
                           .replace('Enhanced bullet point:', '')
                           .replace('Improved bullet point:', '')
                    )
                    cleaned_item = escape_latex(cleaned_item)
                    formatted_items.append(f"      \\resumeItem{{{cleaned_item}}}")
                
                replacement = f"{section_match.group(1)}\n"
                replacement += "\n".join(formatted_items)
                replacement += f"\n{section_match.group(3)}"
                
                result = result[:section_match.start()] + replacement + result[section_match.end():]
    
    return result

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
        for token in tokens if token.isalpha() and token not in stop_words and len(token) > 2
    ]
    unique_keywords = list(dict.fromkeys(keywords))
    technical_keywords = identify_technical_keywords(text, unique_keywords)
    return technical_keywords

def identify_technical_keywords(text: str, keywords: List[str]) -> List[str]:
    """Identify technical keywords using NER and TF-IDF."""
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

def analyze_with_ai(text: str) -> List[str]:
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert technical recruiter specializing in data science and AI roles. 
                    Your task is to analyze job descriptions and extract key technical skills, tools, and requirements."""
                },
                {
                    "role": "user",
                    "content": f"Analyze this job description and extract key technical requirements:\n\n{text}"
                }
            ],
            temperature=0.2,
            max_tokens=400
        )
        
        result = response.choices[0].message.content
        try:
            parsed_result = json.loads(result)
            if isinstance(parsed_result, dict) and "technical_skills" in parsed_result:
                return parsed_result["technical_skills"]
        except json.JSONDecodeError:
            skills = [skill.strip() for skill in result.split('\n') if skill.strip()]
            return skills[:20]
            
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        return []
    
def optimize_bullet_point(bullet: str, keywords: List[str], optimization_level: int = 5) -> str:
    try:
        level_description = {
            range(1, 4): "Make minimal, subtle changes. Only add keywords where they perfectly fit.",
            range(4, 8): "Make moderate changes, naturally incorporating keywords while maintaining authenticity.",
            range(8, 11): "Make significant changes to maximize keyword presence while keeping content professional."
        }
        level_guide = "Make moderate changes, naturally incorporating keywords while maintaining authenticity."
        for key_range, description in level_description.items():
            if optimization_level in key_range:
                level_guide = description
                break
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert at optimizing resume bullet points.
                    Optimization Level: {optimization_level}/10
                    Strategy: {level_guide}
                    Task: Enhance the bullet point by incorporating the provided keywords naturally.
                    Format: Return only the enhanced bullet point text without any prefix or formatting."""
                },
                {
                    "role": "user",
                    "content": f"Keywords to incorporate: {', '.join(keywords)}\n\nOriginal bullet point: {bullet}"
                }
            ],
            temperature=0.3,
            max_tokens=150
        )
        optimized_bullet = response.choices[0].message.content.strip()
        return optimized_bullet
    except Exception as e:
        print(f"Error in optimize_bullet_point: {str(e)}")
        return bullet

def optimize_resume_sections(latex_code: str, keywords: List[str], optimization_level: int = 5) -> str:
    """Optimize the resume sections by enhancing bullet points with specified keywords."""
    try:
        if not validate_latex(latex_code):
            raise ValueError("Invalid LaTeX: Must start with \\documentclass and end with \\end{document}")
        
        sections = extract_sections(latex_code)
        optimized_sections = {}
        
        for section, items in sections.items():
            optimized_items = []
            for item in items:
                optimized_item = optimize_bullet_point(item, keywords, optimization_level)
                optimized_items.append(optimized_item)
            optimized_sections[section] = optimized_items
        
        optimized_latex = replace_sections(latex_code, optimized_sections)
        return optimized_latex
        
    except Exception as e:
        raise Exception(f"Failed to optimize LaTeX: {str(e)}")

