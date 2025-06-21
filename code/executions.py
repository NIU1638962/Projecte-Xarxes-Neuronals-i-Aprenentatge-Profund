# -*- coding: utf-8 -*- noqa
"""
Created on Sat May  3 15:44:15 2025

@author: Joel Tapia Salvador
"""
import environment

CURRENT_EXECUTION = None

PHASES = ('train', 'validation')

NEXT_WHEN_ERROR = False

NUMBER_EPOCHS = 100

EXECUTIONS = {
    # Experiment 1: Baseline Models with Different Hidden Dimensions
    # Execution 1: Hidden Dimension 256
    'experiment_1_execution_1_hidden_dim_256': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'random_split',
            'parameters': {
                'percentage_test': 0.2,
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 700,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'inver_log',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'baseline',
            'parameters': {
                'hidden_dim': 256,
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.5,
            },
        },
        'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (700)),
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
    # Execution 2: Hidden Dimension 512
    'experiment_1_execution_2_hidden_dim_512': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'random_split',
            'parameters': {
                'percentage_test': 0.2,
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 512,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'inver_log',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'baseline',
            'parameters': {
                'hidden_dim': 512,
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.5,
            },
        },
        'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (512)),
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
    # Experiment 2: Full CNN
    # Execution: Full CNN with Hidden Dimension 128
    'experiment_2_execution': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'random_split',
            'parameters': {
                'percentage_test': 0.2,
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 256,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'inver_log',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'full_cnn',
            'parameters': {
                'hidden_dim': 128,
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.7,
            },
        },
        'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (256)), 
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
    # Experiment 3: ResNet Variants
    # Execution 1: ResNet18
    'experiment_3_execution_1_res_net_18': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'random_split',
            'parameters': {
                'percentage_test': 0.2,
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 700,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'inver_log',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'res_net_18',
            'parameters': {
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.7,
            },
        },
        'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (700)),
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
    # Execution 2: ResNet50
    'experiment_3_executions_2_res_net_50': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'random_split',
            'parameters': {
                'percentage_test': 0.2,
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 512,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'inver_log',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'res_net_50',
            'parameters': {
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.7,
            },
        },
        'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (512)),
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
    # Execution 3: ResNet101
    'experiment_3_executions_3_res_net_101': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'random_split',
            'parameters': {
                'percentage_test': 0.2,
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 512,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'inver_log',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'res_net_101',
            'parameters': {
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.7,
            },
        },
        'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (512)),
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
    # Experiment 4: Fine-tuning ResNet18
    # Execution 1: Fine-tune Layer 4
    'experiment_4_execution_1_layer_4': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'random_split',
            'parameters': {
                'percentage_test': 0.2,
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 700,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'inver_log',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'res_net_18_fine_tune_layer_4',
            'parameters': {
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.7,
            },
        },
        'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (700)),
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
    # Execution 2: Fine-tune Layer 4 Last Block
    'experiment_4_execution_2_layer_4_last_block': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'random_split',
            'parameters': {
                'percentage_test': 0.2,
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 700,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'inver_log',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'res_net_18_fine_tune_layer_4_last_block',
            'parameters': {
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.7,
            },
        },
        'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (700)),
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
    # Experiment 5: Fine-tuning ResNet18 Top Level
    # Execution 1: Fine-tune Layer 4 Top Level
    'experiment_5_execution_1_layer_4_top_level': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'levels_split',
            'parameters': {
                'percentage_test': 0.2,
                'selected_levels': [0],
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 700,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'inver_log',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'res_net_18_fine_tune_layer_4',
            'parameters': {
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.7,
            },
        },
        'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (700)),
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
    # Execution 2: Fine-tune Layer 4 Last Block Top Level
    'experiment_5_execution_2_layer_4_last_block_top_level': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'levels_split',
            'parameters': {
                'percentage_test': 0.2,
                'selected_levels': [0],
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 700,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'inver_log',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'res_net_18_fine_tune_layer_4_last_block',
            'parameters': {
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.7,
            },
        },
        'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (700)),
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
    # Experiment 6: Multi-head ResNet18
    # Execution 1: Multi-head Equal Weights
    'experiment_6_execution_1_multihead_equal': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'levels_split',
            'parameters': {
                'percentage_test': 0.2,
                'selected_levels': [],
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 700,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'none',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'res_net_18_multi_head',
            'parameters': {
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.7,
            },
        },
       'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
                1,
                1,
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (700)),
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1/4,
                1/4,
                1/4,
                1/4,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
    # Execution 2: Multi-head Ratio Weights
    'experiment_6_execution_2_multihead_ratio': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'levels_split',
            'parameters': {
                'percentage_test': 0.2,
                'selected_levels': [],
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 600,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'ratio',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'res_net_18_multi_head',
            'parameters': {
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.7,
            },
        },
       'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
                1,
                1,
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (600)),
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1/4,
                1/4,
                1/4,
                1/4,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
    # Execution 3: Multi-head Inv Log Weights
    'experiment_6_execution_3_multihead_inv_log': {
        'data_transformation': {
            'name': 'mel_spectrogram_pre_saved',
            'parameters': {
                'sample_rate': 16000,
                'number_frames': 16000 * 29,
                'number_fft_points': 1024,
                'hop_length': 1024,
                'window_length': 512,
                'power': 2.0,
                'number_mel_channels': 128,
                'to_decibels': True,
            },
        },
        'data_split': {
            'name': 'levels_split',
            'parameters': {
                'percentage_test': 0.2,
                'selected_levels': [],
            },
        },
        'dataset':  {
            'name': 'dataset_with_spectograms_pre_saved',
            'parameters': {},
        },
        'dataloader': {
            'name': 'dataloader',
            'parameters': {
                'batch_size': 600,
                'shuffle': True,
                'pin_memory': False,
                'prefetch_factor': 5,
                'persistent_workers': True,
            },
        },
        'weights': {
            'parameters': {
                'method': 'inver_log',
                'epsilon': 1e-6,
            },
        },
        'model': {
            'name': 'res_net_18_multi_head',
            'parameters': {
                'classifier_hidden_layers': [],
                'threshold_method': 'roc_closest',
                'dropout_rate': 0.7,
            },
        },
       'loss': {
            'losses': [
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
                {
                    'name': 'BCEWithLogitsLoss',
                    'parameters': {
                        'reduction': 'mean',
                    },
                },
            ],
            'weights': [
                1,
                1,
                1,
                1,
            ],
        },
        'optimizer': {
            'name': 'Adam',
            'parameters': {
                'lr': 3e-3,
            },
        },
        'learning_rate_schedulers': [
            {
                'name': 'OneCycleLR',
                'parameters': {
                    'max_lr': 3e-3,
                    'total_steps': NUMBER_EPOCHS * environment.math.ceil(environment.math.floor(102221 * 0.8) / (600)),
                    # total_steps = number_epochs * ceil(floor(data_samples * percentage_train) / batch_size)
                    'pct_start': 0.3,
                    'anneal_strategy': 'cos',
                    'div_factor': 25.0,
                    'final_div_factor': 1e4,
                },
                'monitor': None,
                'frequency': 'batch',
            },
        ],
        'metrics_class': {
            'metrics_classes': [
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
                {
                    'name': 'multi_label_metrics',
                    'parameters': {},
                },
            ],
            'weights': [
                1/4,
                1/4,
                1/4,
                1/4,
            ],
        },
        'weight_init_type': '',
        'number_epochs': NUMBER_EPOCHS,
        'bench_mark': False,
        'max_grad_norm': float('inf'),
        'metrics': [
            'false_negative',
            'false_positive',
            'true_negative',
            'true_positive',
            'accuracy',
        ],
        'objective': (
            'accuracy',
            'maximize',
            'validation',
        ),
    },
}
