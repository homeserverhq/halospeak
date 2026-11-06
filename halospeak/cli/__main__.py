"""Entrypoint for the command-line interface.

This file allows the CLI to be run as a package using the command:
`python -m halospeak.cli`
"""

from halospeak.cli.cli import main

if __name__ == "__main__":
    main()
