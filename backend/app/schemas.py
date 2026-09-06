from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class StudyMeta(BaseModel):
    study_id: str = Field(..., pattern=r"^NSS-\d{8}-\d{4}$")
    patient_id: str = Field(..., min_length=1)
    full_name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=130)
    gender: Literal["Male", "Female", "Other", "Prefer not to say"]
    contact_email: Optional[str] = None
    referring_doctor: Optional[str] = None
    hospital: Optional[str] = None
    scan_date: date
    mri_modality: str
    clinical_notes: Optional[str] = None
