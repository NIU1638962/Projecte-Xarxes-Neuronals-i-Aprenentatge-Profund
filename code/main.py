# -*- coding: utf-8 -*- noqa
"""
Created on Sun Apr 27 19:50:06 2025

@author: Joel Tapia Salvador
"""
import api_wandb
import environment
import executions
import utils

from dataloaders import AVAILABLE_DATALOADERS
from datasets import AVAILABLE_DATASETS
from data_processing import get_data
from data_splitting import AVAILABLE_DATA_SPLITS
from data_transformations import AVAILABLE_DATA_TRANSFORMATIONS
from initialize_model import init_model
from train import train


def main():
    data = get_data()

    for executions.CURRENT_EXECUTION, settings in executions.EXECUTIONS.items():
        api_wandb.RUN_NAME = (
            f'{environment.USER}--{environment.LOG_LEVEL_NAME}'
            + f'--{environment.SEED}--{environment.TIME_EXECUTION}'
            + f'--{executions.CURRENT_EXECUTION or "run"}'
        )

        utils.print_message(
            f'{environment.SECTION_LINE}\n'
            + f'Executing: {executions.CURRENT_EXECUTION}'
        )

        settings['dataloader']['parameters']['num_workers'] = environment.NUMBER_WORKERS
        if environment.NUMBER_WORKERS <= 0:
            settings['dataloader']['parameters']['prefetch_factor'] = None

        utils.print_message(
            environment.SEPARATOR_LINE
            + '\nData transformation: '
            + f'{settings["data_transformation"]["name"]}'
        )

        data = AVAILABLE_DATA_TRANSFORMATIONS[
            settings["data_transformation"]["name"]
        ](
            data,
            **settings["data_transformation"]["parameters"],
        )

        utils.print_message(f'Split type: "{settings["data_split"]["name"]}"')

        for (
            fold,
            (data_tracks, list_data_genres, list_genres_information),
        ) in enumerate(
            AVAILABLE_DATA_SPLITS[settings['data_split']['name']](
                data,
                **settings['data_split']['parameters'],
            )
        ):

            utils.print_message(
                environment.SEPARATOR_LINE
                + f'\nFold: {fold + 1}'
            )

            for phase in executions.PHASES:
                for data_genres in list_data_genres:
                    assert (
                        len(data_tracks[phase])
                    ) == (
                        data_genres[phase].shape[0]
                    )

                    utils.print_message(
                        f'Data {phase}: "{data_genres[phase].shape[0]}" tracks'
                        + f' and "{data_genres[phase].shape[1]}" genres.'
                    )

                    del data_genres
                    utils.collect_memory()
                del phase
                utils.collect_memory()

            utils.print_message(
                f'{environment.SECTION * 3}'
                + 'CALCULATING METRICS WEIGHTS'
                + f'{environment.SECTION * 3}'
            )

            list_genres_fold = []

            for data_genres in list_data_genres:

                genres_fold = environment.torch.empty(
                    (0, (next(iter(data_genres.values()))).shape[1]),
                    dtype=environment.torch.float32,
                )

                for genres_tensor in data_genres.values():
                    genres_fold = environment.torch.cat([
                        genres_fold,
                        genres_tensor,
                    ])

                    del genres_tensor
                    utils.collect_memory()

                list_genres_fold.append(genres_fold)

                del genres_fold
                utils.collect_memory()

            del data_genres
            utils.collect_memory()

            list_weights = utils.compute_weights(
                list_genres_fold,
                **settings['weights']['parameters']
            )

            del list_genres_fold
            utils.collect_memory()

            for (
                index,
                weights,
            ) in zip(
                range(len(settings['loss']['losses'])),
                list_weights,
            ):
                settings[
                    'loss'
                ]['losses'][index]['parameters']['pos_weight'] = weights
                settings[
                    'metrics_class'
                ]['metrics_classes'][index]['parameters']['weights'] = weights

                del index, weights
                utils.collect_memory()

            del list_weights
            utils.collect_memory()

            utils.print_message(
                f'{environment.SECTION * 3}'
                + 'CREATING DATASETS'
                + f'{environment.SECTION * 3}'
            )

            utils.print_message(
                f'Dataset type: "{settings["dataset"]["name"]}"'
            )

            datasets = {
                phase: AVAILABLE_DATASETS[settings['dataset']['name']](
                    data_tracks[phase],
                    [data_genres[phase] for data_genres in list_data_genres],
                    list_genres_information,
                    **settings['dataset']['parameters'],
                ) for phase in executions.PHASES
            }

            utils.print_message(
                '\n'.join(
                    f'Length dataset {phase}: {len(datasets[phase])}' for phase in executions.PHASES
                )
            )

            sample_input, sample_target = datasets[executions.PHASES[0]][0]

            list_number_classes = [sample.shape[0] for sample in sample_target]

            settings[
                'model'
            ]['parameters']['input_dimensions'] = sample_input.shape
            settings[
                'model'
            ]['parameters']['number_classes'] = list_number_classes
            for index, number_classes in zip(
                range(len(settings['metrics_class']['metrics_classes'])),
                list_number_classes,
            ):
                settings[
                    'metrics_class'
                ][
                    'metrics_classes'
                ][index]['parameters']['number_classes'] = number_classes

                del index, number_classes
                utils.collect_memory()

            del sample_input, sample_target, list_number_classes
            utils.collect_memory()

            utils.print_message(
                'Input dimensions: ['
                + ", ".join(
                    str(dimension) for dimension in settings["model"]["parameters"]["input_dimensions"]
                )
                + ']\nNumber classes: '
                + f'{settings["model"]["parameters"]["number_classes"]}'
            )

            utils.print_message(
                f'{environment.SECTION * 3}'
                + 'CREATING DATALOADERS'
                + f'{environment.SECTION * 3}'
            )

            utils.print_message(
                f'Dataloader type: "{settings["dataloader"]["name"]}"'
            )

            dataloaders = {
                phase: AVAILABLE_DATALOADERS[settings['dataloader']['name']](
                    datasets[phase],
                    **settings['dataloader']['parameters'],
                ) for phase in executions.PHASES
            }

            utils.print_message(
                '\n'.join(
                    f'Batches dataloaders {phase}: {len(dataloaders[phase])}' for phase in executions.PHASES
                )
            )

            utils.print_message(
                f'{environment.SECTION * 3}'
                + 'INITIALIZING MODEL'
                + f'{environment.SECTION * 3}'
            )

            (
                model,
                losses_functions,
                optimizer,
                metrics_classes,
                learning_rate_schedulers,
            ) = init_model(
                model_config=settings['model'],
                losses_configs=settings['loss'],
                optimizer_config=settings['optimizer'],
                metrics_classes_configs=settings['metrics_class'],
                learning_rate_schedulers_configs=settings['learning_rate_schedulers'],
                weights_init_type=settings['weight_init_type'],
                bench_mark=settings['bench_mark'],
            )

            utils.print_message(
                f'{environment.SECTION * 3}'
                + 'TRAINING'
                + f'{environment.SECTION * 3}'
            )

            model_parameters = train(
                model=model,
                losses_functions=losses_functions,
                optimizer=optimizer,
                number_epochs=settings['number_epochs'],
                max_grad_norm=settings['max_grad_norm'],
                dataloaders=dataloaders,
                metrics_classes=metrics_classes,
                metrics=settings['metrics'],
                objective=settings['objective'],
                learning_rate_schedulers=learning_rate_schedulers,
            )

            model_file_name = settings.get('model_file_name', None)

            if (
                model_file_name is None
            ) or (
                not model_file_name
            ) or (
                model_file_name == ''
            ):
                model_file_name = f'{api_wandb.RUN_NAME}.pth'

            environment.torch.save(
                model_parameters,
                environment.os.path.join(
                    environment.MODELS_PATH,
                    model_file_name,
                ),
            )

    executions.CURRENT_EXECUTION = None


if __name__ == '__main__':
    environment.init()
    try:
        main()
    except KeyboardInterrupt:
        utils.print_message('Exited manually.')
        environment.finish()
    except:
        environment.finish()
        raise
