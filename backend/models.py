from pydantic import BaseModel
from typing import List, Optional

from pydantic import BaseModel
from typing import List, Optional


class OCRResult(BaseModel):
    text: str
    timestamp: float

class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float

class ReelDocument(BaseModel):
    id: str
    source: str
    source_url: Optional[str]=None

    filename: str

    caption:Optional[str]=None
    transcript:Optional[str]=None
    transcript_segments: List[TranscriptSegment] = []

    ocr_text: List[OCRResult] = []
    visual_description:Optional[str]=None

    comments:List[str]=[]

    created_at:str

