# -*- coding: utf-8 -*- noqa
"""
logger_wandb.py – wrapper WANDB para el proyecto (grupo compartido)
"""
from typing import List
import environment
import executions
import wandb

ENTITY = "grup-1-Xarxes-Neuronals-i-Aprenentatge-Profund"
PROJECT = "fma-genre-classification"
RUN_NAME = 'run'


def start_run(
        cfg: dict,
        tags: List[str] = None,
        job_type: str = None,
        group: str = None,
        notes: str = None,
):
    if not environment.WANDB_LOGGING:
        return None

    return wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=RUN_NAME,
        config=cfg,
        tags=tags,
        job_type=job_type,
        group=group,
        notes=notes,
        save_code=True,
        reinit=True,
    )


def log_metrics(step: int, **metrics):
    if not environment.WANDB_LOGGING:
        return None

    wandb.log(metrics, step=step)


def finish_run(exit_code=0):
    if not environment.WANDB_LOGGING:
        return None

    wandb.finish(exit_code)


def delete_all_runs():
    if not environment.WANDB_LOGGING:
        return None

    api = wandb.Api()
    runs = api.runs(f'{ENTITY}/{PROJECT}')
    for run in runs:
        print(f'Deleting run: {run.name}')
        run.delete()
