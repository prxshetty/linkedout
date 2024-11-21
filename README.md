# linkedout
A web application that updates your resume based on the preferred job description

# Technology Stack
## Backend

- FastAPI - fast Python web framework
- uvicorn - for ASGI
- OpenAIAPI - for llm, might migrate to crew for multi model agentic framework
- spaCy -  nlp/keyword extraction
- PyPDF2 - reading/writing PDF resumes / thinking of converting the resumes first to markdown/ latex file format so llms could have an easier time to edit the resumes.
- python-docx - handling Word documents
- SQLite - Lightweight database for storing user data

## Frontend

- Streamlit
- might later be converted to React/Next.js

## Deployment

- Docker
