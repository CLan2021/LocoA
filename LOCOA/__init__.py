#!/usr/bin/env python

"""
Initialize program
"""

__version__ = "0.0.3"


import sys
from loguru import logger
import LOCOA.locoa_alt

# Logger will show time, function name, and message.
LOGFORMAT = (
    "{time:hh:mm} | {level: <7} | "
    "<b><magenta>{function: <15}</magenta></b> | "
    "<level>{message}</level>"
)

def set_loglevel(loglevel="INFO", quiet=False):
    """
    Set the loglevel for loguru logger. Using 'enable' here as 
    described in the loguru docs for logging inside of a library.
    This sets the level at which logger calls will be displayed 
    throughout the rest of the code.
    """
    config = {}
    config["handlers"] = [{
        "sink": sys.stdout,
        "format": LOGFORMAT,
        "level": loglevel,
        "colorize": True,
    }]
    logger.configure(**config)
    
    if quiet == True:
      logger.disable("LOCOA")
    else:
      logger.enable("LOCOA")