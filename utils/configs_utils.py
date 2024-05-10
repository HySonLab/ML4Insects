import json
import os 
import datetime
from easydict import EasyDict
from copy import deepcopy as dc
import torch
from models.NN import *

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config

def process_config(json_file):
    if not os.path.exists(json_file):
        try:
            config = get_config_from_json(f'./configs/{json_file}')
        except:
            raise RuntimeError(f'{json_file} not found')
    else: 
        config = get_config_from_json(json_file)

    
    time = str(datetime.datetime.now())[:-7]
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.timestamp = time
    return config 