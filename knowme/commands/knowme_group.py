import click
from art import tprint
from knowme.commands.download import download
from knowme.commands.setup import setup


@click.group(name="knowme")
def knowme_group():
    """Root command"""
    pass


def main():
    tprint("KNOW ME")
    knowme_group.add_command(download)
    knowme_group.add_command(setup)
    knowme_group()


if __name__ == "__main__":
    main()
