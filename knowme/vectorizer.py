from typing import Optional

from langchain.text_splitter import TextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class EmbeddingsVectorizer:
    def __init__(self, splitter: TextSplitter, embedding_function: Embeddings):
        """This stores the embedding of the documents into a vector store
        This only considers the RecursiveCharacterTextSplitter for now
        Other TextSplitters might be implemented later

        Parameters
        ----------
        splitter : TextSplitter
            A splitter that splits the documents into smaller pieces of text
        embedding_function: Embeddings
            An Embedding function
        """
        self.splitter = splitter
        self.embedding_function = embedding_function

    def store_embeddings(
        self, texts: list[str], metadatas: Optional[list[dict]]
    ) -> VectorStore:
        """This creates the vector store from the embeddings and then returns it

        Parameters
        ----------
        texts : list[str]
            The list of strings that has been ingested.
        metadatas : Optional[list[dict]]
            Metadata associated with the text

        Returns
        -------
        VectorStore
            Vector store that helps in retrieval
        """
        documents = self.splitter.create_documents(texts=texts, metadatas=metadatas)
        split_documents = self.splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=split_documents, embedding=self.embedding_function
        )

        return vectorstore
