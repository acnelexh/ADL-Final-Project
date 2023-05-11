# utils and miscelaneous functions
import models

def fetch_model(args):
    '''
    build model from args
    '''
    return models.__dict__[args.model](args)