"""
This converts the chains to tools.
These tools can be used by an agent that wishes to work with it
"""

from typing import Optional, Type

from dotenv import load_dotenv
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()


class SiteAnswerInput(BaseModel):
    query: str = Field(description="should be a question from the site")
    session_id: str = Field(
        description="should be a session id where chats in session are remembered"
    )


class CVAnswerInput(BaseModel):
    query: str = Field(description="should be a question from the CV")
    session_id: str = Field(
        description="should be a session id where chats in session are remembered"
    )


class SiteAnswerTool(BaseTool):
    name = "site-answer-tool"
    description = "Answer the questions about the candidate from the website"
    args_schema: Type[BaseModel] = SiteAnswerInput

    def _run(
        self,
        query: str,
        session_id: str,
        run_manager: Optional[CallbackManagerForToolRun],
    ):
        return SiteAnswerTool.chain.chat(query, session_id)

    async def _arun(
        self,
        query: str,
        session_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun],
    ):
        raise NotImplementedError("site-answer-tool does not support async calls")

    def set_chain(self, chain):
        self.chain = chain


class CVAnswerTool(BaseTool):
    name = "cv-answer-tool"
    description = (
        "Answer the questions about the candidate from the CV that is stored in a pdf"
    )
    args_schema: Type[BaseModel] = CVAnswerInput

    def _run(
        self,
        query: str,
        session_id: str,
        run_manager: Optional[CallbackManagerForToolRun],
    ):
        return CVAnswerTool.chain.chat(query, session_id)

    async def _arun(
        self,
        query: str,
        session_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun],
    ):
        raise NotImplementedError("site-answer-tool does not support async calls")

    def set_chain(self, chain):
        self.chain = chain
