from typing import Optional

from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStore


class KnowmeChain:
    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: VectorStore,
    ):
        """This is a retrieval based chat model to know more about a person
        This retrieves information from different vector stores.

        Parameters
        ----------
        llm : BaseChatModel
            You can use any large language model
            for this
        vector_store: VectorStore
            This is the vector store that embeds all the documents
        """
        self.llm = llm
        self.vector_store = vector_store

        self.search_type = "similarity"
        self.top_k_retrieval = 6

        # COnvert the vector store as retriever
        self.retriever = self.vector_store.as_retriever(
            search_type=self.search_type, search_kwargs={"k": self.top_k_retrieval}
        )

        # create the session store. These ar elike cookies and session
        # for persisting a given chat
        self.session_store = {}

        # Refer to the chat history to rewrite the query from the user
        self.contextual_history_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        # The chat_history holds the history of dialogues from previous turns
        self.contextual_rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextual_history_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.history_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextual_rewrite_prompt
        )

        # Create the chat chain here
        # This needs to have the `context` variable to
        # fill the retrieved documents
        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # This is the normal llm with a prompt
        self.chain = create_stuff_documents_chain(self.llm, self.chat_prompt)

        # This rewrites the history with a new prompt to be fed to the previous chain
        self.rag_chain = create_retrieval_chain(self.history_retriever, self.chain)

        # This remembers the message history with a given session store
        self.knowme_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def get_session_history(self, session_id) -> BaseChatMessageHistory:
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()

        return self.session_store[session_id]

    def chat(self, user_input: str, session_id: str):
        """Return the response from the chat agent

        Parameters
        ----------
        user_input : str
            The user typed input
        session_id : str
            The session id that the input belongs to
        """

        return self.knowme_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id},
            },
        )["answer"]
