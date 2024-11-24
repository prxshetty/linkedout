import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict
import re
import spacy
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import os
from dotenv import load_dotenv
from difflib import get_close_matches
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

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
    section_patterns = {
        'experience': r'\\section{Experience}(.*?)\\resumeSubHeadingListEnd',
        'projects': r'\\section{Projects}(.*?)\\resumeSubHeadingListEnd',
        'technical_skills': r'\\section{Technical Skills}(.*?)\\end{itemize}',
    }
    
    for key, pattern in section_patterns.items():
        match = re.search(pattern, latex_code, re.DOTALL | re.IGNORECASE)
        if match:
            section_content = match.group(1).strip()
            sections[key].append(section_content)
    
    return sections

def replace_sections(latex_code: str, optimized_sections: Dict[str, List[str]]) -> str:
    result = latex_code
    
    for section, contents in optimized_sections.items():
        for content in contents:
            pattern = re.compile(r'(\\section{' + re.escape(section.capitalize()) + r'})(.*?)(\\resumeSubHeadingListEnd)', re.DOTALL | re.IGNORECASE)
            match = pattern.search(latex_code)
            if match:
                optimized_content = content
                latex_code = latex_code.replace(match.group(2), optimized_content)
    
    return latex_code
def modify_quotes(text: str) -> str:
    text = text.replace('"', r'``')
    text = text.replace("'", r"''")
    return text

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

def analyze_with_ai(text: str, num_keywords: int = 20) -> List[str]:
    try:
        response = client.chat.completions.create(
            model='gpt-4',
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert ATS (Applicant Tracking System) and HR professional specializing in technical hiring.
                    Your task is to extract EXACTLY {num_keywords} keywords from the job description that would be most relevant for HR screening.
                    
                    Focus on:
                    1. Core technical skills (programming languages, frameworks, tools)
                    2. Industry-standard technologies and platforms
                    3. Required technical certifications
                    4. Key technical methodologies and practices
                    
                    DO NOT include:
                    - Generic soft skills (like "team player", "communication")
                    - Basic computer skills (like "Microsoft Office")
                    - Non-technical requirements (like "Bachelor's degree")
                    - Company-specific tools or systems
                    
                    Format your response as a JSON object:
                    {{
                        "keywords": [
                            "exactly {num_keywords} technical keywords here"
                        ]
                    }}
                    
                    IMPORTANT: The keywords array MUST contain EXACTLY {num_keywords} items."""
                },
                {
                    "role": "user",
                    "content": f"""Extract exactly {num_keywords} technical keywords from this job description that would help a candidate's resume pass ATS screening and impress HR managers:

{text}"""
                }
            ],
            temperature=0.2,
            max_tokens=800
        )
        
        result = response.choices[0].message.content
        try:
            parsed_result = json.loads(result)
            if "keywords" in parsed_result:
                keywords = parsed_result["keywords"]
                if len(keywords) > num_keywords:
                    keywords = keywords[:num_keywords]
                elif len(keywords) < num_keywords:
                    return analyze_with_ai(text, num_keywords)
                return keywords
            
        except json.JSONDecodeError:
            skills = [
                skill.strip() 
                for skill in result.split('\n') 
                if skill.strip() and not any(x in skill.lower() for x in [
                    "degree", "year", "experience", "team", "communication"
                ])
            ]
            if len(skills) >= num_keywords:
                return skills[:num_keywords]
            return analyze_with_ai(text, num_keywords)
            
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        return []
    
def optimize_section(section: str, keywords: List[str], optimization_level: int) -> str:
    try:
        prompt = f"""
You are an expert at optimizing resume sections written in LaTeX. Your task is to enhance the content within the following LaTeX section by incorporating the provided keywords naturally. 

Please preserve all LaTeX commands and formatting. Only modify the text within the \\\resumeSubheading and \\\resumeItem commands as necessary to improve clarity, impact, and keyword optimization based on the given keywords and optimization level.

### Optimization Level: {optimization_level}/10
### Keywords: {', '.join(keywords)}

### Original Section:
{section}

### Optimized Section:
"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at optimizing resume sections written in LaTeX."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        optimized_section = response.choices[0].message.content.strip()
        return optimized_section
    
    except Exception as e:
        print(f"Error in optimize_section: {e}")
        return section

def optimize_resume_sections(latex_code: str, keywords: List[str], optimization_level: int = 5) -> str:
    try:
        if not validate_latex(latex_code):
            raise ValueError("Invalid LaTeX: Must start with \\documentclass and end with \\end{document}")
        
        sections = extract_sections(latex_code)
        optimized_sections = {}
        
        for section, contents in sections.items():
            optimized_contents = []
            for content in contents:
                optimized_content = optimize_section(content, keywords, optimization_level)
                optimized_contents.append(optimized_content)
            optimized_sections[section] = optimized_contents
        
        optimized_latex = replace_sections(latex_code, optimized_sections)
        return optimized_latex
        
    except Exception as e:
        raise Exception(f"Failed to optimize LaTeX: {str(e)}")