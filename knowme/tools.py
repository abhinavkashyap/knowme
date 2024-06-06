# Just construct and define different tools in this place from typing import Optional, Type

from typing import Optional, Type
from langchain_core.tools import BaseTool

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from knowme.load_chains import load_site_answer_chain, laod_cv_answer_chain
from pydantic import BaseModel, Field


site_answer_chain = load_site_answer_chain(
    notion_folderpath="/Users/abhinavkashyap/abhi/projects/knowme/abhinav_notion",
    embedding_store_directory="/Users/abhinavkashyap/abhi/projects/knowme/notion_site_store",
)

cv_answer_chain = laod_cv_answer_chain(
    cv_filepath="/Users/abhinavkashyap/abhi/projects/knowme/CV.pdf",
    embedding_store_directory="/Users/abhinavkashyap/abhi/projects/knowme/cv_vectorstore",
)


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
        return site_answer_chain.chat(query, session_id)

    async def _arun(
        self,
        query: str,
        session_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun],
    ):
        raise NotImplementedError("site-answer-tool does not support async calls")


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
        return cv_answer_chain.chat(query, session_id)

    async def _arun(
        self,
        query: str,
        session_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun],
    ):
        raise NotImplementedError("site-answer-tool does not support async calls")
