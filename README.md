# XN-Project title music genres classification

To execute our work go to [code/final_submission_main.ipynb](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/final_submission_main.ipynb), is a Python Notebook with all the settings of our models and de logic necessary that clicking "Run All" will allow you to train our models adn get the results in Weight & Bias. It may require changing set up of where the data is in your local machine and your Weight & Bias account.

Our report of the project is [Final Report.pdf](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/Final%20Report.pdf)

# Data

We used the [GitHub - mdeff/fma: FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma?tab=readme-ov-file) for this project. In concrete we used the [fma_metadata](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) and [fma_large](https://os.unil.cloud.switch.ch/fma/fma_large.zip) datasets. For the correct working of the program both folder must be inside the same folder on the same level. 

If you are running the notebook change the path to data to your path where it says.

If you are running the code from [code/main.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/main.py) change the path to your data folder in [code/environment.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/environment.py).

# Code

This code was designed to be modular, in the file [code/executions.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/executions.py) there is a variable called `EXECUTIONS` that is a dictionary of dictionary, each entry of that dictionary is the name of an execution and its associated configuration. You can change the ones already there or delete them and create new ones. Here are the parameters of the configuration:

- `data_transformation`: `name` to reference to the dictionary `AVAILABLE_DATA_TRANSFORMATIONS` in [code/data_transformations.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/data_transformations.py). If you want to add a new one, make a new function in that file and add it to the dictionary as the  other. `parameters` that the referenced function recives, except `data` which is already passed by the code.
- `data_split`: `name` to reference to the dictionary `AVAILABLE_DATA_SPLITS` in [code/data_splitting.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/data_splitting.py). If you want to add a new one, make a new function in that file and add it to the dictionary as the other. `parameters` that the referenced function recives, except `data` which is already passed by the code.
- `dataset`: `name` to reference to the dictionary `AVAILABLE_DATASETS` in [code/datasets.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/datasets.py). If you want to add a new one, make a new class in that file and add it to the dictionary as the other. `parameters` that the referenced class recives, except `spectrogram_files_paths`, `list_tracks_genres` and `genres_information` which are already passed by the code.
- `dataloader`: `name` to reference to the dictionary `AVAILABLE_DATALOADERS` in [code/dataloaders.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/dataloaders.py). If you want to add a new one, make a new class in that file and add it to the dictionary as the other. `parameters` that the referenced class recives, except `dataset` which is already passed by the code.
- `weights`: `parameters` that recives the function `compute_weights` in [code/utils.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/utils.py), except `list_label_tensor` which is already passed by the code.
- `model`: `name` to reference to the dictionary `AVAILABLE_MODELS` in [code/models.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/models.py). If you want to add a new one, make a new class in that file and add it to the dictionary as the other. `parameters` that the referenced class recives, except `input_dimensions` and `number_classes` which are already passed by the code.
- `loss`: Dictionary with list of `losses` to apply to the outputs; `name` to reference to the dictionary `AVAILABLE_LOSSES` in [code/available_losses.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/available_losses.py). If you want to add a new one, make a new class in that file and add it to the dictionary as the other. `parameters` that the referenced class recives, except `pos_weight` which is already passed by the code. And list of `weights` wich are float numbers that represent the weight to give to each loss in the same order they appear in the `losses` list.
- `optimizer`: `name` to reference to the dictionary `AVAILABLE_OPTIMIZERS` in [code/available_optimizers.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/available_optimizers.py). If you want to add a new one, make a new class in that file and add it to the dictionary as the other. `parameters` that the referenced class recives, except `params` which is already passed by the code.
- `learning_rate_schedulers`: List of dictionaries with `name` to reference to the dictionary `AVAILABLE_LEARNING_RATE_SCHEDULERS` in [code/available_learning_rate_schedulers.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/available_learning_rate_schedulers.py). If you want to add a new one, make a new class in that file and add it to the dictionary as the other. `parameters` that the referenced class recives. `monitor` name of the metric that will be used to monitor if the scheduler requires one, if not, set to `None`. `frequency` when the scheduler will be steped, some of the schedulers require every `batch` other every `epoch`.
- `metrics_class`: Dictionary with list of `metrics_classes` to apply to the outputs; `name` to reference to the dictionary `AVAILABLE_LOSSES` in [code/metrics.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/metrics.py). If you want to add a new one, make a new class in that file and add it to the dictionary as the other. `parameters` that the referenced class recives, except `weights` and `number_classes` which are already passed by the code. And list of `weights` wich are float numbers that represent the weight to give to each metric in the same order they appear in the `metrics_classes` list.
- `weight_init_type`: Method to use to initialize the weights and bias of the model, found in the function `initialize_weights` of [code/parameters.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/parameters.py)
- `number_epochs`: Number of epochs.
- `bench_mark`: If to use torch bech mark (more slow).
- `max_grad_norm` If there is exploding gradient the gradient will be clippled to this values. Is a float inclusing `inf` if you don't want to clip the gradient.
- `metrics`: metrics of the `metrics_class` that will be calculated. All available can be found in [code/metrics.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/metrics.py). More metric to calculate more overhead between epochs.
- `objective`: Tuple with the name of the `metrics`, `maximize` or `minimize`, `train` or `validation` that will determine which is the best model of what epoch of them all.

The files:

- [code/api_wandb.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/api_wandb.py): Contain the code to upload runs to Weight & Bias, including the set up of the group and project.

- [code/available_learning_rate_schedulers.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/available_learning_rate_schedulers.py): Contains all the available schedulers that can be used.

- [code/available_losses.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/available_losses.py): Contains all the available losses that can be used.

- [code/available_optimizers.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/available_optimizers.py): Contains all available optimizers that can be used.

- [code/data_processing.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/data_processing.py): Contains all the functions that process the dataset to be used, this includes verifying integrity reading it and preparing the strecture that will be used trhough the code.

- [code/data_splitting.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/data_splitting.py): Contain all the functions that can be used to split the dataset.

- [code/data_transformations.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/data_transformations.py): Contains all the functions that are used to transform the input. For example from waveform to mel-spectogram.

- [code/dataloaders.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/dataloaders.py): Contains all dataloaders that can be used.

- [code/datasets.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/datasets.py): Contains all the code of all datasets that can be used.

- [code/environment.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/environment.py): Contains all the code that loads all modules, sets up all environement variables, logging system, checks all paths exists and if not creates them and also checks that the version of all the modules is the correct one.

- [code/executions.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/executions.py): Contains all the settings for each execution of a run to train a model.

- [code/final_submission_main.ipynb](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/final_submission_main.ipynb) Notebook for easy execution of the code to train executions.

- [code/initialize_model.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/initialize_model.py): Code that initializes the model, losses, optimizers, schedulers and metrics for the train.

- [code/main.py](code/main.py): Code containing all main logic of the code, can be run, same as [code/final_submission_main.ipynb](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/final_submission_main.ipynb). 

- [code/metrics.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/metrics.py): Code containing the classes that calculate the metrics during the train loop.

- [code/models.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/models.py): Code containing all the models used.

- [code/parameters.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/parameters.py): Code to initialize the parameters of a model. It only iniitlaizes the parametes with `requires_grad` set to `True` and `should_initialize` set to `True`.

- [code/train.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/train.py): Code containing the training loop, the epoch loop and the class to calculate the best model of the loop.

- [code/utils.py](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/code/utils.py): General purpose code used all around the different files that fits different purposes. 

# Graphics

In [graphics](https://github.com/XarNeu-EngDades/project24-25-01/tree/main/graphics) we store all images that we use in our Report.

# Models

In [models](https://github.com/XarNeu-EngDades/project24-25-01/tree/main/models) is where the best models of each execution are saved as `.pth` file.

# Pickles

In [pickles](https://github.com/XarNeu-EngDades/project24-25-01/tree/main/pickles) is where all dumps of memory are saved that allow faster executions without overhead time, of for example, checking the dataset or calculating splits.

# Scripts

In [scripts](https://github.com/XarNeu-EngDades/project24-25-01/tree/main/scripts) are all scripts used to make more agile the execution process in the Linux server.

# Test

In [test](https://github.com/XarNeu-EngDades/project24-25-01/tree/main/test) are all the test that the automatic GitHub Action performs in every push or merge.

# Environemnt

The files [environment.yml](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/environment.yml) and [requirements.txt](https://github.com/XarNeu-EngDades/project24-25-01/blob/main/requirements.txt) contain respectively the Conda and Pip  modules to install with the versions so the code runs.

## Contributors

- Joel Tapia Salvador (Joel.TapiaS@uab.cat)

- Alejandro Jaime Cabrera ()

- Almoujtaba Kharrat (1639431@uab.cat)

- Carles Galletto Carner ()

Xarxes Neuronals i Aprenentatge Profund
Grau de Data Engineering , 
UAB, 2025
