from pathlib import Path
from typing import Tuple


class NotionIngestor:
    def __init__(self, folder_path: str):
        """_summary_

        Parameters
        ----------
        folder_path : str
            Path where notion page has been exported
            The page exported by notion is .zip folder
            Unzip the folder first
        """
        self.folder_path = Path(folder_path)

    def ingest(self) -> Tuple[list[str], list[dict]]:
        """Reads the documents.

        Returns
        -------
        Tuple[list[str], list[dict]]
            Text from one notion md page is one string
            Every dictionary contains meta information about the document
        This will be used to create documents when SplittingText.
        Read create_documents from TextSplitter class in langchain
        """
        # Ingest all the documents that has been read
        # This also performs some kind of cleaning of the information

        # Read .md files in all the
        markdown_files = self.folder_path.glob("**/*.md")

        all_data = []
        metadata = []

        for filepath in markdown_files:
            with open(filepath) as fp:
                data = fp.read()
                all_data.append(data)
                metadata.append({"filename": str(filepath)})

        return all_data, metadata
