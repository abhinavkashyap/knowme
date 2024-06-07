import os

import click
from dotenv import load_dotenv
from rich.console import Console

from knowme.utils.general_utils import unzip_file
from knowme.utils.google_drive_downloader import GoogleDriveDownloader

load_dotenv()

console = Console()


@click.group()
def download():
    pass


@download.command()
def cv():
    """Download the CV from google drive. This is a sample CV that you can use"""
    with console.status("Downloading the CV"):
        project_dir = f"{os.path.expanduser(os.environ['PROJECT_DIR'])}"
        destination_file = f"{project_dir}/data/CV.pdf"

        downloader = GoogleDriveDownloader()

        # The file_id corresponds to Abhinav's cv
        downloader.download_file_from_google_drive(
            file_id="1DRhcHz5j22xrLhIERpl7dhbTsaXiDTH9", destination=destination_file
        )

        console.print("[green] Downloaded the sample CV file")


@download.command()
def website():
    """Download the sample website from google drive. This is a sample CV that you can use"""

    project_dir = f"{os.path.expanduser(os.environ['PROJECT_DIR'])}"
    destination_file = f"{project_dir}/data/abhinav_notion.zip"

    downloader = GoogleDriveDownloader()

    # The file_id corresponds to Abhinav's cv
    downloader.download_file_from_google_drive(
        file_id="1u2JI7pgDBwoydHE4CtWOm7hjAnB-dS8Z", destination=destination_file
    )

    unzip_file(filepath=destination_file, destination_dir=f"{project_dir}/data")
    console.print("[green] Downloaded the sample website file")


if __name__ == "__main__":
    download()
