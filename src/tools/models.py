"""
Shared data models for all tools
"""
from pydantic import BaseModel
from typing import List, Optional

class Slot(BaseModel):
    """Time slot model - shared across all time-related tools"""
    start_iso: str
    end_iso: str
    label: str
    confidence: float = 0.8

