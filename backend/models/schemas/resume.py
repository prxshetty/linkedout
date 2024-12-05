from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ResumeBase(BaseModel):
    content : str
    format : str

class ResumeCreate(ResumeBase):
    pass

class Resume(ResumeBase):
    id : str
    optimized_content : Optional[str] = None
    keywords : List[str]
    created_at : datetime
    user_id : str

    class Config:
        from_attributes = True