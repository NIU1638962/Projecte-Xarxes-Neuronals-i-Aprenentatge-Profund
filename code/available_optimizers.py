# -*- coding: utf-8 -*- noqa
"""
Created on Wed Mar 26 18:46:15 2025

@author: Joel Tapia Salvador
"""
import environment

AVAILABLE_OPTIMIZERS = {
    'Adadelta': environment.torch.optim.Adadelta,
    'Adagrad': environment.torch.optim.Adagrad,
    'Adam': environment.torch.optim.Adam,
    'Adamax': environment.torch.optim.Adamax,
    'AdamW': environment.torch.optim.AdamW,
    'ASGD': environment.torch.optim.ASGD,
    'LBFGS': environment.torch.optim.LBFGS,
    'NAdam': environment.torch.optim.NAdam,
    'RAdam': environment.torch.optim.RAdam,
    'RMSprop': environment.torch.optim.RMSprop,
    'Rprop': environment.torch.optim.Rprop,
    'SGD': environment.torch.optim.SGD,
    'SparseAdam': environment.torch.optim.SparseAdam,
}
