from knowme.tools import SiteAnswerTool, CVAnswerTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from langchain.agents import AgentExecutor, create_tool_calling_agent
from typing import Optional
import time


class KnowMeAgent:
    def __init__(self, openai_model: Optional[str] = "gpt-4", verbose=True):
        self.openai_model = openai_model
        self.site_answer_tool = SiteAnswerTool()
        self.cv_answer_tool = CVAnswerTool()
        self.tools = [self.site_answer_tool, self.cv_answer_tool]
        self.verbose = verbose
        self.llm = ChatOpenAI(
            model=self.openai_model,
            temperature=0,
            verbose=self.verbose,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You will answer questions from the user about a candidate. 
            The answer might be present in their Notion Website or it might be present in their CV 
            The CV is usually in the pdf format. Your job is to chose the either the site-answer-tool or cv-answer-tool to answer 
            the question from. If the answer is not available in the site-answer-tool use the cv-answer-tool.""",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=self.verbose
        )

    def chat(self, user_input: str, session_id: str):
        """The agent executor reoutes the query to the appropriate
        tool and answers the different questions. The signature matches
        the one that is used in the knowme_chain.py

        Parameters
        ----------
        user_input : str
            The input from the user
        session_id : str
            The session id to be passed to the tools
            This helps in maintaining the history and remembering
            the context
        """
        output = self.agent_executor.invoke(
            {"input": f"{user_input}. Use session_id={session_id} for agent calls "}
        )
        return output["output"]

    def chat_stream(self, user_input: str, session_id: str):
        # TODO: There is no current standard way to do the streaming
        # of the output for the agent_executor. Instead we make this a
        # generator that can be useful for the st.write_stream()
        # or similar methods for showing in the UI
        answer = self.chat(user_input, session_id)

        def generator():
            for word in answer.split():
                yield {"answer": f"{word} "}
                time.sleep(0.05)

        return iter(generator())
