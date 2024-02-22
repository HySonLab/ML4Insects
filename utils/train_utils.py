from models import *

def get_architecture(arch):
    if arch == 'mlp':
        return MLP()
    elif arch == 'cnn1d':
        return CNN1D()
    elif arch == 'cnn2d':
        return CNN2D()
    elif arch == 'ResNet':
        return ResNet()


def get_dataset_group(name):
    if name == 'BCOA':
        return ['BCOA-Wheat', 'day2run2', 'reutday3', 'reutwheat', 'reutwheat2', 
                'reutwheatday', 'reutwheatrun3', 'wheatday2run2', 'wheatday2run3', 'wheatrun']
    elif name == 'zt':
        return ['0zt','8zt','16zt']
    elif name == 'cannabis':
        return ['non-viruliferous-hemp', 'non-viruliferous-potato','viruliferous-hemp', 'viruliferous-potato']
    elif name == 'Anuradha': # BCOA maybe?
        return ['Anuradha16', 'Anuradha27', 'Anuradha42', 'Anuradha57', 'Anuradha75', 'Anuradha84', 'Anuradha107']
    elif name == 'wheatrnai': # BCOA maybe too?
        return ['wheatrnai']
    elif name == 'sorghum': #sorghum
        return ['sorghumaphid']
    elif name == 'ArabidopsisGPA':
        return ['ArabidopsisGPA']
