import os
from typing import Optional

from dotenv import load_dotenv
from langchain.text_splitter import TextSplitter
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


from knowme.embedder import ChromaEmbedder
from knowme.ingest import NotionIngestor, CVIngestor
from knowme.knowme_chain import KnowmeChain

# Load the environment variables
load_dotenv()


def load_site_answer_chain(
    notion_folderpath: str,
    embedding_store_directory: str,
    embedding_function: Optional[Embeddings] = None,
    splitter: Optional[TextSplitter] = None,
    openai_model: Optional[str] = "gpt-4",
):
    """This loads the site answer chain, with some default decisions made by the code

    Parameters
    ----------
    notion_folderpath : str
        Folderpath where the information about the site is stored
    embedding_store_directory : str
        Location where the embeddings will be stored
    embedding_function : Optional[Embeddings], optional
        The embedding function to use to perform the embeddings, by default None
    splitter : Optional[TextSplitter], optional
        A splitter that splits the documents into smaller pieces of text, by default None
    openai_mdoel : Optional[str], optional
        The name of the openai model, by default "gpt4"
    """

    if embedding_function is None:
        embedding_function = OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )

    if splitter is None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )

    ingestor = NotionIngestor(folder_path=notion_folderpath, splitter=splitter)
    documents = ingestor.ingest()
    store = ChromaEmbedder(
        embedding_function=embedding_function,
        embedding_store_directory=embedding_store_directory,
    )
    chromastore = store.store_embeddings(documents)
    llm = ChatOpenAI(model=openai_model, api_key=os.environ["OPENAI_API_KEY"])
    chain = KnowmeChain(llm, chromastore)
    return chain


def load_cv_answer_chain(
    cv_filepath: str,
    embedding_store_directory: str,
    embedding_function: Optional[Embeddings] = None,
    splitter: Optional[TextSplitter] = None,
    openai_model: Optional[str] = "gpt-4",
):
    """This loads the site answer chain, with some default decisions made.
    This is a convenience method that can be used to load the chain

    Parameters
    ----------
    cv_filepath : str
        Folderpath where the information about the site is stored
    embedding_store_directory : str
        Location where the embeddings will be stored
    embedding_function : Optional[Embeddings], optional
        The embedding function to use to perform the embeddings, by default None
    splitter : Optional[TextSplitter], optional
        A splitter that splits the documents into smaller pieces of text, by default None
    openai_mdoel : Optional[str], optional
        The name of the openai model, by default "gpt4"
    """

    if embedding_function is None:
        embedding_function = OpenAIEmbeddings(
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )

    if splitter is None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )

    ingestor = CVIngestor(filename=cv_filepath, splitter=splitter)
    documents = ingestor.ingest()
    store = ChromaEmbedder(
        embedding_function=embedding_function,
        embedding_store_directory=embedding_store_directory,
    )
    chromastore = store.store_embeddings(documents)
    llm = ChatOpenAI(model=openai_model, api_key=os.environ["OPENAI_API_KEY"])
    chain = KnowmeChain(llm, chromastore)
    return chain
