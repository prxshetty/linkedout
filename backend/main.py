from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from backend.analyze import analyze_job_description
from backend.nlp import extract_keywords

class JobDescription(BaseModel):
    description: str

app = FastAPI()

@app.post("/analyze")
async def analyze_job(job: JobDescription):
    result = analyze_job_description(job.description)
    return result