# -*- coding: utf-8 -*- noqa
"""
Created on Thu Jan 23 22:28:45 2025

@author: Joel Tapia Salvador
"""
import ast
import collections
import csv
import gc
import hashlib
import importlib.util
import json
import locale
import logging
import math
import multiprocessing
import os
import pickle
import re
import sys
import traceback
import types
import warnings

from copy import deepcopy
from datetime import datetime, timezone
from time import time

# Modules used
import iterstrat
import iterstrat.ml_stratifiers
import matplotlib
import numpy
import pandas
import psutil
import seaborn
import sklearn
import sklearn.metrics
import soundfile
import torch
import torchaudio
import torchvision
import wandb

from sklearn.exceptions import UndefinedMetricWarning

# Time
TIME_EXECUTION = datetime.now(timezone.utc).strftime(
    '%Y-%m-%d--%H-%M-%S-%f--%Z'
)

# Seed
SEED = 0

# Log level
LOG_LEVEL = logging.INFO
LOG_LEVEL_NAME = logging.getLevelName(LOG_LEVEL)
WANDB_LOGGING = True

# Python version
PYTHON_VERSION = '{0}.{1}.{2}'.format(
    *sys.version_info.__getnewargs__()[0][:3]
)

# Platform
PLATFORM = sys.platform.lower()

# User
USER = ''

if PLATFORM == 'win32':  # Windows
    USER = os.getenv('USERNAME')
else:  # Unix-like platforms
    USER = os.getenv('USER')

# Paths
CODE_PATH = os.path.dirname(os.path.abspath(__file__))

PROJECT_PATH = os.path.dirname(CODE_PATH)

DATA_PATH = os.path.join(PROJECT_PATH, 'data')
LOGS_PATH = os.path.join(PROJECT_PATH, 'logs')
MODELS_PATH = os.path.join(PROJECT_PATH, 'models')
PICKLES_PATH = os.path.join(PROJECT_PATH, 'pickles')
REQUIREMENTS_PATH = deepcopy(PROJECT_PATH)
TORCH_DUMPS_PATH = os.path.join(PROJECT_PATH, 'torch_dumps')

WANDB_LOGS_PATH = os.path.join(LOGS_PATH, 'wandb')

if USER == '':
    DATA_PATH = ''
    LOGS_PATH = ''
    MODELS_PATH = ''
    PICKLES_PATH = ''
    REQUIREMENTS_PATH = ''

elif USER == 'JoelT':
    DATA_PATH = os.path.join('D:', 'projecte-XNAP', 'data')
    PICKLES_PATH = os.path.join('D:', 'projecte-XNAP', 'pickles')
    TORCH_DUMPS_PATH = os.path.join('D:', 'projecte-XNAP', 'torch_dumps')

elif USER == 'EDXN01':
    DATA_PATH = os.path.join(os.sep, 'home', 'datasets', 'FreeMusicArchive')


# Set WANDB log directory
os.environ["WANDB_DIR"] = WANDB_LOGS_PATH


# CPU cores
NUMBER_PHYSICAL_PROCESSORS = psutil.cpu_count(logical=False)
NUMBER_LOGICAL_PROCESSORS = psutil.cpu_count(logical=True)

# Number CPU workers
# if PLATFORM in ['linux', 'darwin']:
#     multiprocessing.set_start_method('fork', force=True)
#     NUMBER_WORKERS = deepcopy(NUMBER_LOGICAL_PROCESSORS)
# else:
#     multiprocessing.set_start_method('spawn', force=True)
#     NUMBER_WORKERS = 0

multiprocessing.set_start_method('spawn', force=True)
NUMBER_WORKERS = deepcopy(NUMBER_LOGICAL_PROCESSORS)
# NUMBER_WORKERS = 0

# Torch CUDA and device
CUDA_AVAILABLE = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if CUDA_AVAILABLE else 'cpu')

# Prints
try:
    TERMINAL_WIDTH = os.get_terminal_size().columns
except OSError as error:
    TERMINAL_WIDTH = 32

SECTION = '='
SECTION_LINE = SECTION * TERMINAL_WIDTH
SEPARATOR = '-'
SEPARATOR_LINE = SEPARATOR * TERMINAL_WIDTH
MARKER = '*'

EXECUTION_INFORMATION = (
    SECTION_LINE
    + '\nENVIRONMENT INFO\n'
    + SEPARATOR_LINE
    + f'\nTime execution: {TIME_EXECUTION}'
    + f'\nSeed: {SEED}'
    + f'\nLog level: {LOG_LEVEL_NAME}'
    + f'\nWandb logging: {WANDB_LOGGING}'
    + f'\nPython version: {PYTHON_VERSION}'
    + f'\nPlatform: {PLATFORM}'
    + f'\nUser: {USER}'
    + f'\nPath to project: "{PROJECT_PATH}"'
    + f'\nPath to code: "{CODE_PATH}"'
    + f'\nPath to data: "{DATA_PATH}"'
    + f'\nPath to logs: "{LOGS_PATH}"'
    + f'\nPath to models: "{MODELS_PATH}"'
    + f'\nPath to pickles: "{PICKLES_PATH}"'
    + f'\nPath to requirements: "{REQUIREMENTS_PATH}"'
    + f'\nPath to torch dumps: "{TORCH_DUMPS_PATH}"'
    + f'\nPath to wandb logs: "{WANDB_LOGS_PATH}"'
    + f'\nPhysical Processors: {NUMBER_PHYSICAL_PROCESSORS}'
    + f'\nLogical Processors: {NUMBER_LOGICAL_PROCESSORS}'
    + f'\nNumber workers: {NUMBER_WORKERS}'
    + f'\nCuda available: {CUDA_AVAILABLE}'
    + f'\nTorch device: {str(TORCH_DEVICE).replace(":", " ")}'
    + f' ({torch.cuda.get_device_properties(TORCH_DEVICE).name})' if CUDA_AVAILABLE else ''
)


def init():
    # Create paths in case they do not exist
    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)
    os.makedirs(CODE_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)
    os.makedirs(PICKLES_PATH, exist_ok=True)
    os.makedirs(REQUIREMENTS_PATH, exist_ok=True)
    os.makedirs(TORCH_DUMPS_PATH, exist_ok=True)
    os.makedirs(WANDB_LOGS_PATH, exist_ok=True)

    # Logging set up
    logging.basicConfig(
        filename=os.path.join(
            LOGS_PATH,
            f'{TIME_EXECUTION}--{PLATFORM}--{USER}--{LOG_LEVEL_NAME}.log',
        ),
        filemode='w',
        level=LOG_LEVEL,
        force=True,
        format='[%(asctime)s] %(levelname)s:\n\tModule: "%(module)s"\n\t' +
        'Function: "%(funcName)s"\n\tLine: %(lineno)d\n\tLog:\n\t\t%(message)s\n',
    )

    logging.info(EXECUTION_INFORMATION.replace('\n', '\n\t\t'))
    print(EXECUTION_INFORMATION)

    # Test versions
    if PYTHON_VERSION != '3.8.20':
        ERROR = f'Python 3.8.20 not found ({PYTHON_VERSION}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise SystemError(ERROR)
    if iterstrat.__version__ != '0.1.9':
        ERROR = f'Module iterstrat 0.1.9 not found ({iterstrat.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)
    if matplotlib.__version__ != '3.7.3':
        ERROR = (
            f'Module matplotlib 3.7.3 not found ({matplotlib.__version__}).'
        )
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)
    if seaborn.__version__ != '0.13.2':
        ERROR = f'Module seaborn 0.13.2 not found ({seaborn.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)
    if numpy.__version__ != '1.24.4':
        ERROR = f'Module numpy 1.24.4 not found ({numpy.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)
    if pandas.__version__ != '2.0.3':
        ERROR = f'Module pandas 2.0.3 not found ({pandas.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)
    if psutil.__version__ != '6.0.0':
        ERROR = f'Module psutil 6.0.0 not found ({psutil.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)
    if sklearn.__version__ != '1.3.2':
        ERROR = f'Module sklearn 1.3.2 not found ({sklearn.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)
    if soundfile.__version__ != '0.12.1':
        ERROR = f'Module soundfile 0.12.1 not found ({soundfile.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)
    if not torch.__version__.startswith('2.4.1'):
        ERROR = f'Module torch 2.4.1 not found ({torch.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)
    if not torchaudio.__version__.startswith('2.4.1'):
        ERROR = (
            f'Module torchaudio 2.4.1 not found ({torchaudio.__version__}).'
        )
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)
    if not torchvision.__version__.startswith('0.19.1'):
        ERROR = (
            f'Module torchvision 0.19.1 not found ({torchvision.__version__}).'
        )
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)
    if wandb.__version__ != '0.16.6':
        ERROR = f'Module wandb 0.16.6 not found ({wandb.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)

    # Setting modules
    pandas.set_option('display.max_columns', None)
    warnings.filterwarnings(
        "ignore", category=UndefinedMetricWarning)


def finish():
    logging.shutdown()
