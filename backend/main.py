from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from backend.analyze import analyze_job_description, analyze_job_description_with_ai
from backend.nlp import extract_keywords, analyze_with_ai

class JobDescription(BaseModel):
    description: str

app = FastAPI()

@app.post("/analyze")
async def analyze_job(job: JobDescription):
    result = analyze_job_description(job.description)
    return result


@app.post("/analyze_ai")
async def analyze_job_with_ai(job: JobDescription):
    result = analyze_job_description_with_ai(job.description)
    return result