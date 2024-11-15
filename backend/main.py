from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from backend.analyze import analyze_job_description, analyze_job_description_with_ai
from backend.nlp import extract_keywords, analyze_with_ai, optimize_latex_content

class JobDescription(BaseModel):
    description: str

class LatexOptimization(BaseModel):
    latex_code : str
    keywords : list[str]
    optimization_level : int = 5

app = FastAPI()

@app.post("/analyze")
async def analyze_job(job: JobDescription):
    result = analyze_job_description(job.description)
    return result


@app.post("/analyze_ai")
async def analyze_job_with_ai(job: JobDescription):
    result = analyze_job_description_with_ai(job.description)
    return result

@app.post("/optimize_latex")
async def optimize_latex(data: LatexOptimization):
    try:
        optimized_latex = optimize_latex_content(data.latex_code, data.keywords, data.optimization_level)
        return {
            "status": "success",
            "optimized_latex": optimized_latex
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
