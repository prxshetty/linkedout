from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class JobBase(BaseModel):
    title : str
    description : str
    company : Optional[str] = None
    url : Optional[str] = None

class JobCreate(JobBase):
    id: str
    keywords : List[str]
    created_at : datetime
    user_id : Optional[str] = None

    class Config:
        from_attributes = True

    