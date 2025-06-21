# -*- coding: utf-8 -*- noqa
"""
Created on Wed Mar 26 18:33:49 2025

@author: Joel Tapia Salvador
"""
import environment

AVAILABLE_LOSSES = {
    'BCELoss': environment.torch.nn.BCELoss,
    'BCEWithLogitsLoss': environment.torch.nn.BCEWithLogitsLoss,
    'CosineEmbeddingLoss': environment.torch.nn.CosineEmbeddingLoss,
    'CrossEntropyLoss': environment.torch.nn.CrossEntropyLoss,
    'CTCLoss': environment.torch.nn.CTCLoss,
    'GaussianNLLLoss': environment.torch.nn.GaussianNLLLoss,
    'HuberLoss': environment.torch.nn.HuberLoss,
    'HingeEmbeddingLoss': environment.torch.nn.HingeEmbeddingLoss,
    'KLDivLoss': environment.torch.nn.KLDivLoss,
    'L1Loss': environment.torch.nn.L1Loss,
    'MSELoss': environment.torch.nn.MSELoss,
    'MarginRankingLoss': environment.torch.nn.MarginRankingLoss,
    'MultiLabelMarginLoss': environment.torch.nn.MultiLabelMarginLoss,
    'MultiLabelSoftMarginLoss': environment.torch.nn.MultiLabelSoftMarginLoss,
    'MultiMarginLoss': environment.torch.nn.MultiMarginLoss,
    'NLLLoss': environment.torch.nn.NLLLoss,
    'PoissonNLLLoss': environment.torch.nn.PoissonNLLLoss,
    'SmoothL1Loss': environment.torch.nn.SmoothL1Loss,
    'SoftMarginLoss': environment.torch.nn.SoftMarginLoss,
    'TripletMarginLoss': environment.torch.nn.TripletMarginLoss,
    'TripletMarginWithDistanceLoss': environment.torch.nn.TripletMarginWithDistanceLoss,
}
