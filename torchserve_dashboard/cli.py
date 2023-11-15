from typing import Any
import click
import streamlit.cli
from streamlit.cli import configurator_options
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command(context_settings=dict(ignore_unknown_options=True,
                                     allow_extra_args=True))
@configurator_options
@click.argument("args", nargs=-1)
@click.pass_context
def main(ctx: click.Context, args: Any, **kwargs: Any):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'dash.py')
    logger.log(f"Arguments:\nArgs:\n{args}\nKwargs:\n{kwargs}")
    ctx.forward(streamlit.cli.main_run, target=filename, args=args)
