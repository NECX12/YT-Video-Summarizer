from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class GraphState(BaseModel):
    video_url : str = Field(description= "The URL for the youtube video")
    video_id : Optional[str] = Field(default=None, description= "The Id of the youtrube video")
    transcript : Optional[str] = Field(default=None, description= "The transcript of the youtube video")
    summary : Optional[str] = Field(default=None, description= "The summary of the youtube video")
    keyword : Optional[List[str]] = Field(default=None, description = "The keyword extracted from the youtube video")
    video_suggestions : Optional[str] = Field(default=None, description= "List of suggested videos based on youtube videos")
    questions : Optional[str] = Field(default=None, description= "Questions generated from the youtube video")
    next_steps : Optional[str] = Field(default=None, description= "Next step based on the youtube video")
