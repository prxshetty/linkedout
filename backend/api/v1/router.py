from fastapi import APIRouter
from .endpoints import jobs, resume, auth

router = APIRouter()

router.include_router(auth.router, prefix = "/auth", tags = ["auth"])
router.include_router(jobs.router, prefix = "/jobs", tags = ["jobs"])
router.include_router(resume.router, prefix = "/resume", tags = ["resume"])