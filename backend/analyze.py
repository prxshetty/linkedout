from backend.nlp import extract_keywords
from backend.nlp import analyze_with_ai

def analyze_job_description(job_description: str):
    try:
        keywords = extract_keywords(job_description)
        return {
            "keywords" : keywords,
            "status" : "success"
        }
    except Exception as e:
        print(f"Error analyzing job description : {e}")
        return {
            "keywords" : [],
            "status" : "error",
            "message" : str(e)
        }
def analyze_job_description_with_ai(job_description):
    try:
        ai_keywords = analyze_with_ai(job_description)
        if not ai_keywords:
            return {
                "keywords" : [],
                "status" : "error",
                "message" : "No keywords were extracted. Please check the input text."
            }
        return {
            "keywords" : ai_keywords,
            "status" : "success"
        }
    except Exception as e:
        print(f"Error analyzing job description with AI: {e}")
        return {
            "keywords" : [],
            "status" : "error",
            "message" : str(e)
        }