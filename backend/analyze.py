from backend.nlp import extract_keywords, analyze_with_ai, optimize_resume_sections
import json

def analyze_job_description(job_description: str):
    try:
        keywords = extract_keywords(job_description)
        response = {
            "keywords": keywords,
            "status": "success"
        }
        print(f"Analysis Response: {json.dumps(response, indent=2)}")
        print("=== Analysis Complete ===\n")
        return response
    except Exception as e:
        error_response = {
            "keywords": [],
            "status": "error",
            "message": str(e)
        }
        print(f"Error in analysis: {json.dumps(error_response, indent=2)}")
        return error_response

def analyze_job_description_with_ai(job_description):
    try:
        print("\n=== Starting AI Analysis ===")
        ai_keywords = analyze_with_ai(job_description)
        if not ai_keywords:
            error_response = {
                "keywords": [],
                "status": "error",
                "message": "No keywords were extracted. Please check the input text."
            }
            print(f"AI Analysis Error: {json.dumps(error_response, indent=2)}")
            return error_response

        response = {
            "keywords": ai_keywords,
            "status": "success"
        }
        print(f"AI Analysis Response: {json.dumps(response, indent=2)}")
        print("=== AI Analysis Complete ===\n")
        return response
    except Exception as e:
        error_response = {
            "keywords": [],
            "status": "error",
            "message": str(e)
        }
        print(f"Error in AI analysis: {json.dumps(error_response, indent=2)}")
        return error_response
    
def optimize_resume_sections_wrapper(latex_code: str, keywords: list, optimization_level: int = 5) -> str:
    optimized_latex, processing_time = optimize_resume_sections(latex_code, keywords, optimization_level)
    return {
        "optimized_latex": optimized_latex,
        "processing_time" : round(processing_time, 2)
    }