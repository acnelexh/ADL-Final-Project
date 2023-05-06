# utils and miscelaneous functions
import models
from pathlib import Path

def fetch_model(args):
    return models.__dict__[args.model](args)