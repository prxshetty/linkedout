from backend.nlp import extract_keywords

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