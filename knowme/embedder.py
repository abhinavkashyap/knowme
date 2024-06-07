from typing import Optional
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pathlib import Path
from langchain_core.documents import Document


class ChromaEmbedder:
    def __init__(
        self,
        embedding_function: Embeddings,
        embedding_store_directory: Optional[str],
    ):
        """This stores the embedding of the documents into a vector store
        This only considers the RecursiveCharacterTextSplitter for now
        Other TextSplitters might be implemented later

        Parameters
        ----------
        embedding_function: Embeddings
            An Embedding function
        embeddings_store_directory: Optional[str]
            If this is provided, then the `store_embeddings` will
            store the emebeddings in this store. Further calls to the
            `store_embeddings` will retrieve the store from this
            directory
        """
        self.embedding_function = embedding_function
        self.embedding_store_directory = embedding_store_directory

    def store_embeddings(self, documents: list[Document]) -> VectorStore:
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
            Vector store that can be used to retrieve documents
        """

        # If the user has provided a directory to store
        # and there is no such directory, then we create the embeddings
        # and store them. otherwise, you just load the embeddings
        if (
            self.embedding_store_directory
            and not Path(self.embedding_store_directory).is_dir()
        ):
            print("creating the embedding store")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=self.embedding_store_directory,
            )
        else:
            return Chroma(
                persist_directory=str(self.embedding_store_directory),
                embedding_function=self.embedding_function,
            )

        return vectorstore
