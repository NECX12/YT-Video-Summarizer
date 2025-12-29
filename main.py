from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import YouTubeSearchTool
from langgraph.graph import StateGraph, START, END
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

llm = ChatGroq(model="llama-8b-instant", temperature=0.8)

class GraphState(BaseModel):
    video_url : str = Field(description= "The URL for the youtube video")
    video_id : Optional[str] = Field(default=None, description= "The Id of the youtrube video")
    transcript : Optional[str] = Field(default=None, description= "The transcript of the youtube video")
    summary : Optional[str] = Field(default=None, description= "The summary of the youtube video")
    keyword : Optional[List[str]] = Field(default=None, description = "The keyword extracted from the youtube video")
    video_suggestions : Optional[str] = Field(default=None, description= "List of suggested videos based on youtube videos")
    questions : Optional[str] = Field(default=None, description= "Questions generated from the youtube video")
    next_steps : Optional[str] = Field(default=None, description= "Next step based on the youtube video")


class ExtractVideoID(BaseModel):
    video_id : str = Field(description= "The Id of the youtube video")


def extract_video_id(state: GraphState):
    video_url = state.video_url
    template = PromptTemplate(
        template=
        """Extract the video ID from this youtube URL: {video_url}
        Return only the video ID""",
        input_variables = ["video_url"]
    )
    llm_with_structured_output = llm.with_structured_output(ExtractVideoID)
    chain = template | llm_with_structured_output
    result = chain.invoke({"video_url": video_url})
    return {"video_id": result.video_id}

def extract_transcript(state: GraphState):
    video_id = state.video_id
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    transcript_text = " "
    for snip in fetched_transcript:
       transcript_text += " " + snip.text
    return {"transcript": transcript_text}

def summarize_transcript(state: GraphState):
    transcript = state.transcript
    template = PromptTemplate(
        template ="""
        Summarize the following transcript in a concise manner: {transcript}
    """,
    input_variables = [transcript])
    chain = template | llm
    result = chain.invoke({"transcript": transcript})
    return {"summary": result.content}

def generate_questions(state: GraphState):
    summary = state.summary
    template = PromptTemplate(
        template = """
        From the summary content given generate 5 questions: {summary}
        """,
        input_variables = ["summary"]
    )
    chain = template | llm
    result = chain.invoke({"summary": summary})
    return {"questions": result.content}

def next_steps(state: GraphState):
    summary = state.summary
    template = PromptTemplate(
        template = """
        What next steps would you suggest based on the summary given: {summary}
        For instance, if the video is about the fundamentals of supervised machine learning
        you can get into using the titanic dataset and explaining how it relates to supervised learning
        """,
        input_variables = ["summary"]
    )
    chain = template | llm
    result = chain.invoke({"summary": summary})
    return {"next_steps": result.content}


def find_keyword(state: GraphState):
    transcript = state.transcript
    template = PromptTemplate(
        template = """
        Extract the most relevant keyword from the following transcript:
        {transcript}
        """,
        input_variables = ["transcript"]
    )
    chain = template | llm
    result = chain.invoke({"transcript": transcript})
    return {"keyword": result.content}

def video_suggestion(state: GraphState):
    keyword = state.keyword
    tool = YouTubeSearchTool()
    video_suggestions = tool.invoke(keyword)
    return {"video_suggestions": video_suggestions}

builder = StateGraph(GraphState)

