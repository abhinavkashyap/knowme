from typing import Optional

from langchain.text_splitter import TextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pathlib import Path


class ChromaEmbedder:
    def __init__(
        self,
        splitter: TextSplitter,
        embedding_function: Embeddings,
        embedding_store_directory: Optional[str],
    ):
        """This stores the embedding of the documents into a vector store
        This only considers the RecursiveCharacterTextSplitter for now
        Other TextSplitters might be implemented later

        Parameters
        ----------
        splitter : TextSplitter
            A splitter that splits the documents into smaller pieces of text
        embedding_function: Embeddings
            An Embedding function
        embeddings_store_directory: Optional[str]
            If this is provided, then the `store_embeddings` will
            store the emebeddings in this store. Further calls to the
            `store_embeddings` will retrieve the store from this
            directory
        """
        self.splitter = splitter
        self.embedding_function = embedding_function
        self.embedding_store_directory = embedding_store_directory

    def store_embeddings(
        self, texts: Optional[list[str]] = None, metadatas: Optional[list[dict]] = None
    ) -> VectorStore:
        """This creates the vector store from the embeddings and then returns it
        Either the mebeddings are created based on the texts and metadatas that
        you pass or loaded from the directory
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

        if (
            self.embedding_store_directory
            and not Path(self.embedding_store_directory).is_dir()
        ):
            assert texts is not None, "Pass the texts to be embedded into the store"
            documents = self.splitter.create_documents(texts=texts, metadatas=metadatas)
            split_documents = self.splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(
                documents=split_documents,
                embedding=self.embedding_function,
                persist_directory=self.embedding_store_directory,
            )
        else:
            return Chroma(
                persist_directory=str(self.embedding_store_directory),
                embedding_function=self.embedding_function,
            )

        return vectorstore
