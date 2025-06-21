# -*- coding: utf-8 -*- noqa
"""
Created on Fri May  2 14:56:05 2025

@author: Joel Tapia Salvador
"""
from typing import Dict, List, Tuple, Union

import api_wandb
import environment
import executions
import utils


def train(
        model: environment.torch.nn.Module,
        losses_functions: Dict[str, List[Union[environment.torch.nn.Module, float]]],
        optimizer: environment.torch.optim.Optimizer,
        number_epochs: int,
        max_grad_norm: int,
        dataloaders: Dict[str, environment.torch.utils.data.DataLoader],
        metrics_classes,
        metrics: List[str] = [],
        objective: Tuple[str] = ('loss', 'minimize', 'validation'),
        learning_rate_schedulers=[],
):
    time_log = {}

    metrics_log = {
        name_metric: {
            phase: [] for phase in executions.PHASES
        } for name_metric in metrics if getattr(
            getattr(metrics_classes['metrics_classes'][0], name_metric, False),
            'is_metric',
            False,
        )
    }

    metrics_log['loss'] = {}
    metrics_log['total_grad_norm'] = {}

    for phase in executions.PHASES:
        metrics_log['loss'][phase] = []
        metrics_log['total_grad_norm'][phase] = {
            type_total_grad_norm: environment.wandb.Table(
                columns=["epoch", "batch", "value"],
            ) for type_total_grad_norm in ('gotten', 'clipped')
        }
        time_log[phase] = []

    best_model = BestModel(model.state_dict(), objective[1])

    tags = [
        environment.TIME_EXECUTION,
        executions.CURRENT_EXECUTION or 'Unknown Execution',
        environment.USER or 'Unknown User',
    ]

    notes = (
        environment.EXECUTION_INFORMATION
        + f'\nExecuting: "{executions.CURRENT_EXECUTION}"'
    )

    api_wandb.start_run(
        cfg=executions.EXECUTIONS[executions.CURRENT_EXECUTION],
        tags=tags,
        job_type=environment.LOG_LEVEL_NAME,
        group=str(environment.SEED),
        notes=notes,
    )

    try:
        for epoch in range(1, number_epochs + 1):
            utils.print_message(
                f'{environment.SEPARATOR * 3}'
                + f'Epoch {epoch}/{number_epochs}'
                + f'{environment.SEPARATOR * 3}'
            )

            for phase in executions.PHASES:
                utils.print_message(
                    f'{environment.MARKER}'
                    + f'Phase: {phase}'
                    + f'{environment.MARKER}'
                )

                (
                    model,
                    mean_loss,
                    epoch_time,
                    metrics_classes,
                    total_grad_norm,
                ) = __epoch(
                    model=model,
                    losses_functions=losses_functions,
                    optimizer=optimizer,
                    dataloader=dataloaders[phase],
                    learning_rate_schedulers=learning_rate_schedulers,
                    phase=phase,
                    metrics_classes=metrics_classes,
                    max_grad_norm=max_grad_norm,
                )

                utils.print_message('Calculating phase final metrics...')

                metrics_log['loss'][phase].append(mean_loss)
                for type_total_grad_norm, values in total_grad_norm.items():
                    for batch, value in enumerate(values):
                        metrics_log[
                            'total_grad_norm'
                        ][phase][type_total_grad_norm].add_data(
                            epoch,
                            batch,
                            value,
                        )

                        del batch, value
                        utils.collect_memory()

                    del type_total_grad_norm
                    utils.collect_memory()

                time_log[phase].append(epoch_time)

                for metric_name in metrics:
                    metrics_log[metric_name][phase].append(
                        environment.torch.tensor(
                            [
                                weight * metrics_class.__getattribute__(
                                    metric_name
                                )() for metrics_class, weight in zip(
                                    metrics_classes['metrics_classes'],
                                    metrics_classes['weights'],
                                )
                            ]
                        ).sum().item()
                    )

                    del metric_name
                    utils.collect_memory()

                del mean_loss, epoch_time, total_grad_norm, phase
                utils.collect_memory()

                utils.print_message('Calculated phase final metrics.')

            utils.print_message(
                f'{environment.MARKER}'
                + 'Logging epoch metrics'
                + f'{environment.MARKER}'
            )

            epoch_metrics = {}

            epoch_metrics['learning_rate'] = [
                group["lr"] for group in optimizer.param_groups
            ][0]

            for phase in executions.PHASES:

                epoch_metrics[f'loss_{phase}'] = metrics_log['loss'][phase][-1]
                epoch_metrics[f'time_{phase}'] = time_log[phase][-1]

                # ─── CONFUSION MATRICES ─────────────────────
                for m_idx, m_cls in enumerate(metrics_classes['metrics_classes']):
                    # 2×2 global
                    cm2 = m_cls.confusion_matrix_2x2().cpu()
                    fig2, ax2 = environment.matplotlib.pyplot.subplots(
                        figsize=(3, 3))
                    environment.seaborn.heatmap(
                        cm2, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Pos", "Neg"], yticklabels=["Pos", "Neg"],
                        ax=ax2,
                    )
                    ax2.set_title(f"Global 2×2 | {phase} | head {m_idx}")
                    api_wandb.log_metrics(
                        epoch,
                        **{f"cm2_head{m_idx}_{phase}": environment.wandb.Image(fig2)}
                    )
                    environment.matplotlib.pyplot.close(fig2)

                    # L×L multi-label
                    cmL = m_cls.confusion_matrix_ll().cpu()
                    l = cmL.shape[0]
                    figL, axL = environment.matplotlib.pyplot.subplots(
                        figsize=(6, 5))
                    environment.seaborn.heatmap(
                        cmL, annot=False, cmap="Blues",
                        xticklabels=False, yticklabels=False, ax=axL,
                    )
                    axL.set_xlabel("Predicted label")
                    axL.set_ylabel("True label")
                    axL.set_title(f"{l}×{l} | {phase} | head {m_idx}")
                    api_wandb.log_metrics(
                        epoch,
                        **{f"cmL_head{m_idx}_{phase}": environment.wandb.Image(figL)}
                    )
                    environment.matplotlib.pyplot.close(figL)
                # ────────────────────────────────────────────────────────────

                # --- métricas escalares definidas en 'metrics' --------------
                for metric_name in metrics:
                    epoch_metrics[f'{metric_name}_{phase}'] = (
                        metrics_log[metric_name][phase][-1]
                    )
            # ========== FIN del bucle por fases =============================

            # ---- limpiar acumulados para la siguiente época ---------------
            for m_cls in metrics_classes['metrics_classes']:
                m_cls.reset()

            del phase
            utils.collect_memory()

            api_wandb.log_metrics(step=epoch, **epoch_metrics)

            if len(learning_rate_schedulers) > 0:
                utils.print_message(
                    f'{environment.MARKER}'
                    + 'Updating learning rate'
                    + f'{environment.MARKER}'
                )

            __update_learning_rate(
                learning_rate_schedulers,
                metrics_log,
                'epoch',
            )

            utils.print_message(
                f'{environment.MARKER}'
                + f'Results epoch'
                + f'{environment.MARKER}\n'
                + '\n'.join(
                    f'Time {phase}: {time_log[phase][-1]} seconds' for phase in executions.PHASES
                )
                + '\n'
                + '\n'.join(
                    f'Loss {phase}: {metrics_log["loss"][phase][-1]}' for phase in executions.PHASES
                )
            )

            best_model.update(
                epoch,
                time_log[objective[2]][epoch - 1],
                metrics_log[objective[0]][objective[2]][epoch - 1],
                model.state_dict(),
            )

            del epoch, epoch_metrics
            utils.collect_memory()

        api_wandb.log_metrics(
            None,
            **{
                f'{phase}_total_grad_norm_{type_total_grad_norm}': table for phase in executions.PHASES for type_total_grad_norm, table in metrics_log['total_grad_norm'][phase].items()

            }
        )

        utils.print_message(
            f'{environment.SEPARATOR * 3}'
            + f'Best model'
            + f'{environment.SEPARATOR * 3}'
            + f'\nIn epoch: {best_model.epoch_number}\n'
            + f'With {objective[2]} {objective[0]}: {best_model.model_metric}'
            + f'\nTook: {best_model.epoch_time} seconds'
        )
    except KeyboardInterrupt:
        api_wandb.finish_run(0)
        utils.print_message('Remotely stopped.')
    except BaseException as error:
        api_wandb.finish_run(-1)
        utils.print_message(str(error))
        utils.print_error()
        if executions.NEXT_WHEN_ERROR:
            return best_model.model_parameters
        else:
            raise error from error
    else:
        api_wandb.finish_run(0)
        return best_model.model_parameters


def __epoch(
    model: environment.torch.nn.Module,
    losses_functions: Dict[str, List[environment.torch.nn.Module]],
    optimizer: environment.torch.optim.Optimizer,
    dataloader: environment.torch.utils.data.DataLoader,
    learning_rate_schedulers: environment.torch.optim.lr_scheduler.LRScheduler,
    phase: str,
    metrics_classes,
    max_grad_norm: int,
):
    if phase == 'train':
        model.train()

    elif phase == 'validation':
        model.eval()
    else:
        error = (
            f'Unexpected "{phase}" phase. Expected "train" or "validation".'
        )
        environment.logging.error(error.replace('\n', '\n\t\t'))
        raise AttributeError(error)

    total_grad_norm = {
        'gotten': [],
        'clipped': [],
    }

    with environment.torch.set_grad_enabled(phase == 'train'):
        running_epoch_loss = 0.0

        epoch_start_time = environment.time()

        list_all_logits = [
            environment.torch.empty(
                0,
                dtype=environment.torch.float32,
            ).to(environment.TORCH_DEVICE) for i in range(
                len(losses_functions['losses_functions']),
            )
        ]

        list_all_targets = [
            environment.torch.empty(
                0,
                dtype=environment.torch.float32,
            ).to(environment.TORCH_DEVICE) for i in range(
                len(losses_functions['losses_functions']),
            )
        ]

        for batch_index, (inputs, targets) in enumerate(dataloader):
            utils.print_message(
                f'Computing batch: {batch_index + 1}/{len(dataloader)}'
            )

            batch_size = inputs.size(0)
            optimizer.zero_grad()

            inputs = inputs.to(environment.TORCH_DEVICE)
            outputs = model(inputs)

            del inputs
            utils.collect_memory()

            with environment.torch.no_grad():
                results = model.results(outputs)

            targets = [
                target.to(environment.TORCH_DEVICE) for target in targets
            ]
            step_loss = 0
            for (
                loss_function,
                weight,
                output,
                result,
                target,
                metrics_class,
            ) in zip(
                losses_functions['losses_functions'],
                losses_functions['weights'],
                outputs,
                results,
                targets,
                metrics_classes['metrics_classes'],
            ):

                if loss_function.reduction == 'mean':
                    step_loss += (
                        weight * loss_function(output, target) * batch_size
                    )
                elif loss_function.reduction == 'sum':
                    step_loss += weight * loss_function
                else:
                    error = 'Only accepted loss reduction is "mean" or "sum".'
                    environment.logging.error(error.replace('\n', '\n\t\t'))
                    raise AttributeError(error)

                metrics_class.update(result, target)

                del (
                    loss_function,
                    weight,
                    output,
                    result,
                    target,
                    metrics_class,
                )
                utils.collect_memory()

            running_epoch_loss += step_loss.item()

            # utils.print_message(
            #     f"step_loss: {step_loss.item()}, batch_size: {batch_size}"
            # )

            if phase == 'train':
                step_loss.backward()

                total_grad_norm['gotten'].append(
                    environment.torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_grad_norm,
                    )
                )

                total_grad_norm['clipped'].append(
                    environment.torch.norm(
                        environment.torch.stack(
                            [
                                p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None
                            ],
                        ),
                        2,
                    ),
                )

                optimizer.step()

                __update_learning_rate(
                    learning_rate_schedulers,
                    None,
                    'batch',
                )

            elif phase == 'validation':
                total_grad_norm['gotten'].append('None')
                total_grad_norm['clipped'].append('None')

            list_all_logits = [
                environment.torch.cat(
                    [all_loggits, ouput.detach()]
                ) for all_loggits, ouput in zip(list_all_logits, outputs)
            ]

            list_all_targets = [
                environment.torch.cat(
                    [all_targets, target.detach()]
                ) for all_targets, target in zip(list_all_targets, targets)
            ]

            utils.print_message('Computed')

            del step_loss, outputs, results, targets
            utils.collect_memory()

        model.update_threshold(
            list_all_logits,
            list_all_targets,
        )

        del list_all_logits, list_all_targets

        epoch_end_time = environment.time()
        epoch_time = epoch_end_time - epoch_start_time

        # utils.print_message(f"Total running_epoch_loss: {running_epoch_loss}")
        # utils.print_message(f"Dataset size: {len(dataloader.dataset)}")

        mean_epoch_loss = running_epoch_loss / len(dataloader.dataset)

        # utils.print_message(f"Final mean_epoch_loss: {mean_epoch_loss}")

    return model, mean_epoch_loss, epoch_time, metrics_classes, total_grad_norm


def __update_learning_rate(learning_rate_schedulers, metrics_log, point):
    for learning_rate_scheduler in learning_rate_schedulers:
        if learning_rate_scheduler['frequency'] == point:
            if learning_rate_scheduler['monitor'] is None:
                learning_rate_scheduler['learning_rate_scheduler'].step()
            else:
                learning_rate_scheduler['learning_rate_scheduler'].step(
                    metrics_log[learning_rate_scheduler['monitor'][0]][
                        learning_rate_scheduler['monitor'][1]
                    ][-1]
                )

            if point == 'epoch':
                utils.print_message(
                    'Learning rate now is:'
                    + f' {learning_rate_scheduler["learning_rate_scheduler"].get_last_lr()[0]}'
                )


class BestModel():

    __slots__ = (
        '__epoch_number',
        '__epoch_time',
        '__model_metric',
        '__model_parameters',
        '__objective',
    )

    def __init__(
        self,
        starting_parameters: environment.torch.nn.parameter.Parameter,
        objective: str = 'minimize',
    ):
        if objective in ('minimize', 'min', '-'):
            self.__objective = 'minimize'
            initial_metric = float('inf')
        elif objective in ('maximize', 'max', '+'):
            self.__objective = 'maximize'
            initial_metric = float('-inf')
        else:
            error = (
                'Objective is a not supported one.'
                + ' Supported objectives are "minimize" or "maximize".'
                + f' Got "{objective}" instead.'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise ValueError(error)

        self.__set(0, 0.0, initial_metric, starting_parameters)

    def __set(
        self,
        epoch_number: int,
        epoch_time: float,
        model_metric: float,
        model_parameters: environment.torch.nn.parameter.Parameter,
    ):
        if not isinstance(epoch_number, int):
            error = (
                f'"epoch_number" is not an integer, is a {type(epoch_number)}.'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise TypeError(error)

        if not isinstance(epoch_time, float):
            error = (
                f'"epoch_time" is not a float, is a {type(epoch_time)}.'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise TypeError(error)

        if not isinstance(model_metric, float):
            error = (
                f'"model_metric" is not a float, is a {type(epoch_number)}.'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise TypeError(error)

        if not isinstance(
                model_parameters,
                environment.collections.OrderedDict,
        ):
            error = (
                '"model_parameters" is not a Collections Ordered Dictionary'
                + f', is a {type(model_parameters)}.'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise TypeError(error)

        if epoch_number < 0:
            error = '"epoch_number" cannot be negative.'
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise ValueError(error)

        if epoch_time < 0:
            error = '"epoch_time" cannot be negative.'
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise ValueError(error)

        self.__epoch_number = epoch_number
        self.__epoch_time = epoch_time
        self.__model_metric = model_metric
        self.__model_parameters = environment.deepcopy(model_parameters)

    def update(
        self,
        new_epoch_number: int,
        new_epoch_time: float,
        new_model_metric: float,
        new_model_parameters: environment.collections.OrderedDict,
    ) -> bool:
        """
        Update values of best model if the new metric is better than previous.

        Better is controled by the initialized "objective" parameter in the
        object's creation. If set to "minimize" a lesser value will be
        considered better. If set to "maximize" a greater value will be
        considered better.

        The function will deepcopy the parameters if it updates them,
        so there is no need to deepcopy them before passing them to the
        function.

        Parameters
        ----------
        new_epoch_number : Integer
            Number of the epoch where the new model was trained.
        new_epoch_time : Float
            Duration that the epoch took to train the new model.
        new_model_metric : Float
            Value of the comparasing metric valued over the new model.
        new_model_parameters : Collections Ordered Dictionary
            Internal parameters of the new model.

        Raises
        ------
        ValueError
            Any of the funtions parameters has a not allowed value for what it
            represents or the objective has been set to a not recognized mode.

        Returns
        -------
        is_better : Boolean
            Wheter the new model is better than the previous best one.

        """
        if self.__objective == 'minimize':
            is_better = new_model_metric < self.__model_metric
        elif self.__objective == 'maximize':
            is_better = new_model_metric > self.__model_metric
        else:
            error = (
                'Objective is a not supported one.'
                + ' Supported objectives are "minimize" or "maximize".'
                + f' Got "{self.__objective}" instead.'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise ValueError(error)

        if is_better:
            self.__set(
                new_epoch_number,
                new_epoch_time,
                new_model_metric,
                new_model_parameters,
            )

        return is_better

    @property
    def epoch_number(self) -> int:
        """
        Getter for epoch number.

        Returns
        -------
        Integer
            Number of the epoch where the best model was trained.

        """
        return self.__epoch_number

    @property
    def epoch_time(self) -> float:
        """
        Getter for epoch time.

        Returns
        -------
        Float
            Duration that the epoch took to train the best model.

        """
        return self.__epoch_time

    @property
    def model_metric(self) -> float:
        """
        Getter for model metric.

        Returns
        -------
        Float
            Value of the comparasing metric valued over the best model.

        """
        return self.__model_metric

    @property
    def model_parameters(self) -> environment.collections.OrderedDict:
        """
        Getter for model parameters.

        Returns
        -------
        Collections Ordered Dictionary
            Internal parameters of the best model.

        """
        return self.__model_parameters
