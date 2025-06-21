# -*- coding: utf-8 -*- noqa
"""
Created on Thu Mar 27 01:35:33 2025

@author: Joel Tapia Salvador
"""
from typing import Dict, List, Tuple, Union

import environment
import utils

from available_learning_rate_schedulers import AVAILABLE_LEARNING_RATE_SCHEDULERS
from available_losses import AVAILABLE_LOSSES
from available_optimizers import AVAILABLE_OPTIMIZERS
from metrics import AVAILABLE_METRICS_CLASSES, MetricsClass
from models import AVAILABLE_MODELS
from parameters import (
    get_number_of_model_parameters,
    get_parameters_to_update,
    initialize_weights,
)


def init_model(
    model_config: Dict[str, Union[str, Dict[str, str]]],
    losses_configs: Dict[str, List[Dict[str, Union[str, Dict[str, str]]]]],
    optimizer_config: Dict[str, Union[str, Dict[str, str]]],
    metrics_classes_configs: Dict[str, List[Dict[str, Union[str, Dict[str, str]]]]],
    learning_rate_schedulers_configs: List[Dict[str, Union[str, Dict[str, str]]]],
    weights_init_type: str,
    bench_mark: bool = False,
) -> Tuple[
    environment.torch.nn.Module,
    environment.torch.nn.Module,
    environment.torch.optim.Optimizer,
    MetricsClass,
    environment.torch.optim.lr_scheduler.LRScheduler,
]:
    """
    Initialize model, weights, loss and optimizer for the training.

    Parameters
    ----------
    model_type : Torch Module
        Model type.
    model_init_parameters : Dictionary
        Parameters to initilaize the model.
    loss_type : Torch Module or String
        Loss type.
    loss_init_parameters : Dictionary
        Paramters to initialize the loss.
    optimizer_type : Torch Optimizer or String
        Type of optimizer to use.
    optimizer_init_parameters : Dictionary
        Parameters to initialize the optimizer. Does not include the model
        internal parameters, those are obtained by the funtion.
    weights_init_type : String
        Type of weight initialization used. Options:
            - 'xavier_normal'
            - 'xavier_uniform'
            - 'kaiming_normal'
            - 'kaiming_uniform'
            - 'orthogonal'
    bench_mark : Boolean, optional
        Wether activate CUDA benchmark. The default is False.
    manual_seed : Integer or None, optional
        Seed for reproductibity. The default is None.

    Returns
    -------
    model : Torch Module
        Model to be trained.
    loss_function : Torch Module
        Loss function for the training.
    optimizer : Torch Optimizer
        Optimizer for the training.

    """
    # Activate CUDA benchamrks or not (slows down training if activated)
    environment.torch.backends.cudnn.benchmark = bench_mark

    # Seed for reproductibility
    if environment.SEED is not None:
        environment.torch.manual_seed(environment.SEED)

    # -------------------------------------------------------------------------
    # Initialize model
    utils.print_message(f'Model to be trained: "{model_config["name"]}"')
    model = AVAILABLE_MODELS[model_config['name']](
        **model_config['parameters']
    )
    number_of_parameters = get_number_of_model_parameters(model)

    # Initialize weight of the model
    initialize_weights(model, weights_init_type)

    # Get parameters to update (and maybe update) while training
    (
        name_parameters_train,
        parameters_train,
        number_of_parameters_train
    ) = get_parameters_to_update(model)

    utils.print_message(
        'Number of parameters of the model:'
        + f' {number_of_parameters:_}'.replace('_', ' ')
        + '\nNumber of parameters to train:'
        + f' {number_of_parameters_train:_}'.replace('_', ' ')
        + '\nParameters that will be updated during training:'
        + f'\n\t{environment.SEPARATOR} '
        + f'\n\t{environment.SEPARATOR} '.join(name_parameters_train)
    )

    # Move model to device
    model.to(environment.TORCH_DEVICE)

    # -------------------------------------------------------------------------
    # Inicialize loss
    losses = {
        'losses_functions': [],
        'weights': losses_configs['weights'],
    }
    for loss_config in losses_configs['losses']:
        utils.print_message(f'Loss to be used: "{loss_config["name"]}"')
        loss_function = AVAILABLE_LOSSES[loss_config['name']](
            **loss_config['parameters'],
        )

        utils.print_message(
            'Loss function reduction:'
            + f' {loss_function.reduction}'
        )

        losses['losses_functions'].append(loss_function)

        del loss_config, loss_function
        utils.collect_memory()

    # -------------------------------------------------------------------------
    # Inicialize optimizer
    utils.print_message(
        f'Optimizer to be used: "{optimizer_config["name"]}"'
    )
    optimizer = AVAILABLE_OPTIMIZERS[optimizer_config['name']](
        parameters_train,
        **optimizer_config['parameters'],
    )

    # -------------------------------------------------------------------------
    # Initialize metrics class
    metrics = {
        'metrics_classes': [],
        'weights': metrics_classes_configs['weights'],
    }
    for metrics_class_config in metrics_classes_configs['metrics_classes']:
        utils.print_message(
            f'Metrics Class to be used: "{metrics_class_config["name"]}"'
        )

        metrics_class = AVAILABLE_METRICS_CLASSES[metrics_class_config['name']](
            **metrics_class_config['parameters'],
        )

        metrics['metrics_classes'].append(metrics_class)

        del metrics_class_config, metrics_class
        utils.collect_memory()

    # -------------------------------------------------------------------------
    # Initialize learning rate schedulers
    learning_rate_schedulers = []
    utils.print_message(
        'Learning rate schedulers to be used:'
    )
    for learning_rate_scheduler_config in learning_rate_schedulers_configs:
        utils.print_message(
            f'\t{environment.SEPARATOR} '
            + f'"{learning_rate_scheduler_config["name"]}"'
        )

        learning_rate_schedulers.append(
            {
                'learning_rate_scheduler': (
                    AVAILABLE_LEARNING_RATE_SCHEDULERS[
                        learning_rate_scheduler_config['name']
                    ](
                        optimizer,
                        **learning_rate_scheduler_config['parameters']
                    )
                ),
                'monitor': learning_rate_scheduler_config['monitor'],
                'frequency': learning_rate_scheduler_config['frequency']
            }
        )

    return (
        model,
        losses,
        optimizer,
        metrics,
        learning_rate_schedulers,
    )
