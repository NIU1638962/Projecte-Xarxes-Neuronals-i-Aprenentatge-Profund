# -*- coding: utf-8 -*- noqa
"""
Created on Thu Mar 27 01:52:35 2025

@author: Joel Tapia Salvador
"""
from typing import Tuple, List

import environment


def get_number_of_model_parameters(model: environment.torch.nn.Module) -> int:
    """
    Get the number of parameters of a torch model.

    Parameters
    ----------
    model : Torch Module
        Model with the parameters to be counted.

    Returns
    -------
    number_parameters : Integer
        Numbers of parameters in the model.

    """
    number_parameters = 0

    for parameter in list(model.parameters()):
        number_parameters += parameter.numel()

    return number_parameters


def get_parameters_to_update(
        model: environment.torch.nn.Module,
) -> Tuple[List[str], List[environment.torch.nn.parameter.Parameter], int]:
    """
    Get the parameters that will be updated in the training of the model.

    Parameters
    ----------
    model : Torch Module
        Model that will be trained and where the parameters come from.

    Returns
    -------
    name_of_parameters_to_update : List[Strings]
        Name of the parameters that will be updated in the training.
    parameters_to_update : List[Torch Parameter]
        Parameters that will be updated in the training.
    number_of_parameters_to_update : Integer
        NUmber of parameters to update in the training.

    """
    name_of_parameters_to_update = []
    parameters_to_update = []
    number_of_parameters_to_update = 0

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            name_of_parameters_to_update.append(name)
            parameters_to_update.append(parameter)
            number_of_parameters_to_update += parameter.numel()

    return (
        name_of_parameters_to_update,
        parameters_to_update,
        number_of_parameters_to_update,
    )


def initialize_weights(
        model: environment.torch.nn.Module,
        init_type: str = 'xavier_normal',
):
    """
    Initialize the weigths of a torch model using the type given.

    Parameters
    ----------
    model : Torch Module
        Model wich weight are gonna be initalized.
    init_type : String, optional
        Type of initialization used. Options:
            - 'xavier_normal'
            - 'xavier_uniform'
            - 'kaiming_normal'
            - 'kaiming_uniform'
            - 'orthogonal'
        The default is 'xavier_normal'.

    Returns
    -------
    None.

    """
    for module in model.modules():
        if getattr(module, 'should_initialize', False):

            if isinstance(
                module,
                (
                    environment.torch.nn.Conv1d,
                    environment.torch.nn.Conv2d,
                    environment.torch.nn.Conv3d,
                    environment.torch.nn.ConvTranspose1d,
                    environment.torch.nn.ConvTranspose2d,
                    environment.torch.nn.ConvTranspose3d,
                    environment.torch.nn.Linear,
                    environment.torch.nn.Bilinear,
                ),
            ):
                if init_type == 'xavier_normal':
                    environment.torch.nn.init.xavier_normal_(module.weight)
                elif init_type == 'xavier_uniform':
                    environment.torch.nn.init.xavier_uniform_(module.weight)
                elif init_type == 'kaiming_normal':
                    environment.torch.nn.init.kaiming_normal_(
                        module.weight,
                        mode='fan_in',
                        nonlinearity='relu',
                    )
                elif init_type == 'kaiming_uniform':
                    environment.torch.nn.init.kaiming_uniform_(
                        module.weight,
                        mode='fan_in',
                        nonlinearity='relu',
                    )
                elif init_type == 'orthogonal':
                    environment.torch.nn.init.orthogonal_(module.weight)

                if module.bias is not None:
                    environment.torch.nn.init.constant_(module.bias, 0)

            elif isinstance(
                    module,
                    (
                        environment.torch.nn.BatchNorm1d,
                        environment.torch.nn.BatchNorm2d,
                        environment.torch.nn.BatchNorm3d,
                        environment.torch.nn.LayerNorm,
                        environment.torch.nn.GroupNorm,
                        environment.torch.nn.InstanceNorm1d,
                        environment.torch.nn.InstanceNorm2d,
                        environment.torch.nn.InstanceNorm3d,
                    ),
            ):
                environment.torch.nn.init.constant_(module.weight, 1)
                environment.torch.nn.init.constant_(module.bias, 0)

            elif isinstance(
                    module,
                    (
                        environment.torch.nn.LSTM,
                        environment.torch.nn.GRU,
                        environment.torch.nn.RNN,
                    ),
            ):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        if init_type in ('xavier_normal', 'xavier_uniform'):
                            environment.torch.nn.init.xavier_uniform_(param)
                        elif init_type in (
                            'kaiming_normal',
                            'kaiming_uniform',
                        ):
                            environment.torch.nn.init.kaiming_uniform_(param)
                        elif init_type == 'orthogonal':
                            environment.torch.nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        environment.torch.nn.init.constant_(param, 0)

            elif isinstance(module, environment.torch.nn.Embedding):
                environment.torch.nn.init.normal_(module.weight, mean=0, std=1)

            elif isinstance(module, environment.torch.nn.MultiheadAttention):
                if hasattr(
                        module,
                        'in_proj_weight',
                ) and module.in_proj_weight is not None:
                    environment.torch.nn.init.xavier_uniform_(
                        module.in_proj_weight
                    )
                if hasattr(
                        module,
                        'out_proj',
                ) and module.out_proj.weight is not None:
                    environment.torch.nn.init.xavier_uniform_(
                        module.out_proj.weight
                    )
                if hasattr(
                        module,
                        'in_proj_bias',
                ) and module.in_proj_bias is not None:
                    environment.torch.nn.init.constant_(module.in_proj_bias, 0)
                if hasattr(
                        module,
                        'out_proj',
                ) and module.out_proj.bias is not None:
                    environment.torch.nn.init.constant_(
                        module.out_proj.bias, 0)
