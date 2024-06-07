import click
from art import tprint
from knowme.commands.download import download
from knowme.commands.setup import setup


@click.group(name="knowme")
def knowme():
    """This serves as a root command for all the other command."""
    pass


def main():
    tprint("KNOW ME")
    knowme.add_command(download)
    knowme.add_command(setup)
    knowme()


if __name__ == "__main__":
    main()
