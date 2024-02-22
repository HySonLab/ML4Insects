import json
import os 
import datetime
from easydict import EasyDict
from copy import deepcopy as dc
import torch
from models import *

learning_rates = {'zt': 1e-3, 'hemp': 1e-3, 'wheat': 1e-3, 'wheatrnai': 1e-3, 'sorghum': 1e-3}
number_of_epochs = {'zt': 40, 'hemp': 40, 'wheat': 40, 'wheatrnai': 40, 'sorghum': 40}

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
    wd = os.getcwd()
    if not os.path.exists(json_file):
       
        config_path = os.path.join(wd, 'configs')
        if os.path.exists(config_path):
            os.chdir(config_path)
            config = get_config_from_json(json_file)
            os.chdir(wd)
        else:
            raise RuntimeError(f'{json_file} not found')
    else: 
        config = get_config_from_json(json_file)

    
    time = str(datetime.datetime.now())[:-7]
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.timestamp = time
    config.wd = wd
    # if config.exp_name == "10fold_CV":
    #     config.num_epochs = number_of_epochs[config.dataset_name]
    #     config.learning_rate = learning_rates[config.dataset_name]
    
    return config 

def get_architecture_from_config(config):
    if config.arch == 'mlp':
        return MLP()
    elif config.arch == 'cnn2d':
        return CNN2D()
    elif config.arch == 'cnn1d':
        return CNN1D()
    elif config.arch == 'ResNet':
        return ResNet()
    else: 
        raise RuntimeError(f'Architecture {config.arch} is undefined.')