import os

import click
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from knowme.utils.write_env import update_env_file

console = Console()

load_dotenv()


@click.group()
def setup():
    pass


@setup.command()
@click.option(
    "--data-dir-name", type=str, help="The name for your data directory", default="data"
)
@click.option(
    "--stores-dir-name",
    type=str,
    help="""The name for your stores directory. 
    The embeddings will be stored here""",
    default="stores",
)
def directories(data_dir_name, stores_dir_name):
    """Create appropriate directories on repository

    Parameters
    ----------
    data_dir_name : str
        Directory to store all the user related ata
    stores_dir_name : _type_
        Directory that stores all the embeddings of the file
    """
    project_dir = f"{os.path.expanduser('~/.knowme')}"

    # create the data dir and the stores dir

    data_dir = f"{project_dir}/{data_dir_name}"
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        data_dir.mkdir(parents=True)

    stores_dir = f"{project_dir}/{stores_dir_name}"
    stores_dir = Path(stores_dir)

    if not stores_dir.is_dir():
        stores_dir.mkdir(parents=True)

    # update the env file with these values
    update_env_file(
        {
            "DATA_DIR": str(data_dir),
            "STORES_DIR": str(stores_dir),
            "PROJECT_DIR": str(project_dir),
        }
    )
    console.print("[green] Created directories to store data and embeddings")


@setup.command()
@click.option("--api_key", prompt=True, hide_input=True, confirmation_prompt=True)
def openai(api_key):
    """Store the api key in the env file

    Parameters
    ----------
    api_key : str
        OPENAI API KEY that you want stored in the env file
    """
    update_env_file({"OPENAI_API_KEY": api_key})

    console.print("[green] We have stored the api key in your .env file")


if __name__ == "__main__":
    setup()
