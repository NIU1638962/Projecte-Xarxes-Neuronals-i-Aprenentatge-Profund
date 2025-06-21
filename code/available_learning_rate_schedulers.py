# -*- coding: utf-8 -*- noqa
'''
Created on Sat May 17 15:35:56 2025

@author: Joel Tapia Salvador
'''
import environment

AVAILABLE_LEARNING_RATE_SCHEDULERS = {
    'ChainedScheduler': environment.torch.optim.lr_scheduler.ChainedScheduler,
    'ConstantLR': environment.torch.optim.lr_scheduler.ConstantLR,
    'CosineAnnealingLR': environment.torch.optim.lr_scheduler.CosineAnnealingLR,
    'CosineAnnealingWarmRestarts': environment.torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'CyclicLR': environment.torch.optim.lr_scheduler.CyclicLR,
    'ExponentialLR': environment.torch.optim.lr_scheduler.ExponentialLR,
    'LambdaLR': environment.torch.optim.lr_scheduler.LambdaLR,
    'LinearLR': environment.torch.optim.lr_scheduler.LinearLR,
    'MultiStepLR': environment.torch.optim.lr_scheduler.MultiStepLR,
    'MultiplicativeLR': environment.torch.optim.lr_scheduler.MultiplicativeLR,
    'OneCycleLR': environment.torch.optim.lr_scheduler.OneCycleLR,
    'PolynomialLR': environment.torch.optim.lr_scheduler.PolynomialLR,
    'ReduceLROnPlateau': environment.torch.optim.lr_scheduler.ReduceLROnPlateau,
    'SequentialLR': environment.torch.optim.lr_scheduler.SequentialLR,
    'StepLR': environment.torch.optim.lr_scheduler.StepLR,
}
