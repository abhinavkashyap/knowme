from pathlib import Path
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import TextSplitter


class NotionIngestor:
    def __init__(self, folder_path: str, splitter: TextSplitter):
        """_summary_

        Parameters
        ----------
        folder_path : str
            Path where notion page has been exported
            The page exported by notion is .zip folder
            Unzip the folder first
        splitter : TextSplitter
            A splitter that splits the documents into smaller pieces of text
        """
        self.splitter = splitter
        self.folder_path = Path(folder_path)

    def ingest(self) -> list[Document]:
        """Reads the documents.

        Returns
        -------
        list[Document]
            This returns a list of Documents
            These documents will be used to create the vector store
        """
        # Ingest all the documents that has been read
        # This also performs some kind of cleaning of the information

        # Read .md files in all the
        markdown_files = self.folder_path.glob("**/*.md")

        all_data = []
        metadatas = []

        for filepath in markdown_files:
            with open(filepath) as fp:
                data = fp.read()
                all_data.append(data)
                metadatas.append({"filename": str(filepath)})

        documents = self.splitter.create_documents(texts=all_data, metadatas=metadatas)
        split_documents = self.splitter.split_documents(documents)

        return split_documents


class CVIngestor:
    def __init__(self, filename: str, splitter: TextSplitter):
        """This ingests a cv in a pdf format

        Parameters
        ----------
        filename : str
            The name of the file to ingest
        splitter: TextSplitter
            A splitter that splits the documents into smaller pieces of text
        """
        self.filename = filename
        self.splitter = splitter

    def ingest(self) -> list[Document]:
        loader = UnstructuredPDFLoader(self.filename)
        pages: list[Document] = loader.load()
        pages = self.splitter.split_documents(pages)

        return pages
