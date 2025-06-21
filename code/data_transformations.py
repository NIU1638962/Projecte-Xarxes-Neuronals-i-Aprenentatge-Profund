# -*- coding: utf-8 -*- noqa
"""
Created on Mon May 12 13:15:29 2025

@author: Joel Tapia Salvador
"""
from typing import Union

import environment
import utils


def nothing(data):
    return data


def mel_spectrogram_pre_saved(
    data: dict,
    sample_rate: int,
    number_frames: int,
    number_fft_points: int,
    hop_length: int,
    window_length: Union[int, None],
    power: float,
    number_mel_channels: int,
    to_decibels: bool,
):
    data_transformed_paths_path = environment.os.path.join(
        environment.PICKLES_PATH,
        (
            f'{environment.PLATFORM}--{environment.USER}'
            + f'--{environment.LOG_LEVEL_NAME}--{environment.SEED}'
            + f'--mel_spectogram_transformation--{number_frames}'
            + f'--{sample_rate}--{number_fft_points}'
            + f'--{hop_length}--{window_length}'
            + f'--{power}--{number_mel_channels}'
            + f'--{to_decibels}.pickle'
        ),
    )

    if environment.os.path.exists(data_transformed_paths_path):
        data_transformed = utils.load_pickle(data_transformed_paths_path)

    else:

        data_transformed = environment.deepcopy(data)

        data_transformed['tracks'] = []
        data_transformed['genres'] = environment.torch.empty(
            (0, data['genres'].shape[1]),
        )

        directory_path = environment.os.path.join(
            environment.TORCH_DUMPS_PATH,
            'spectograms',
            f'{number_frames}--{sample_rate}--True--{number_fft_points}'
            + f'--{hop_length}--{window_length}--{power}'
            + f'--{number_mel_channels}--{to_decibels}',
        )

        for i in range(len(data['tracks'])):
            utils.print_message(
                f'{environment.SEPARATOR * 3}'
                + f'Loading file {i + 1} / {len(data["tracks"])}'
                + f'{environment.SEPARATOR * 3}'
            )

            audio_file_path = data['tracks'][i]

            spectogram_file_path = utils.remove_file_extension(
                environment.os.path.join(
                    directory_path,
                    environment.os.path.relpath(
                        audio_file_path,
                        start=environment.DATA_PATH,
                    ),
                ),
            ) + '.pt'

            utils.print_message(f'File path: {spectogram_file_path}')

            if environment.os.path.exists(spectogram_file_path):
                utils.print_message('Found.')

            else:
                utils.print_message('Not found.\nCreating it...')

                result = utils.load_audio(
                    file_path=audio_file_path,
                    target_sample_rate=sample_rate,
                    number_frames=number_frames,
                )

                if result is None:
                    utils.print_message(f"[SKIP] Skipping {audio_file_path} due to loading error")
                    continue  # <- saltar al siguiente archivo
                
                waveform, sample_rate = result
                track = utils.normalize_audio(waveform)


                spectrogram = environment.torch.nan_to_num(
                    utils.compute_spectrogram(
                        waveform=track,
                        sample_rate=sample_rate,
                        mel_scale=True,
                        n_fft=number_fft_points,
                        hop_length=hop_length,
                        win_length=window_length,
                        power=power,
                        n_mels=number_mel_channels,
                        to_db=to_decibels
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                environment.os.makedirs(
                    environment.os.path.dirname(spectogram_file_path),
                    exist_ok=True,
                )

                utils.save_torch(spectrogram, spectogram_file_path)

                del track, spectrogram
                utils.collect_memory()

            data_transformed['genres'] = environment.torch.cat([
                data_transformed['genres'],
                data['genres'][i].unsqueeze(0),
            ])

            data_transformed['tracks'].append(spectogram_file_path)

        assert (
            len(data_transformed['tracks'])
        ) == (
            data_transformed['genres'].shape[0]
        ), 'Length of transformed data does not match.'

        utils.save_pickle(data_transformed, data_transformed_paths_path)

    utils.print_message(
        f'Transformed data of "{data_transformed["genres"].shape[0]}" tracks.'
    )

    return data_transformed


AVAILABLE_DATA_TRANSFORMATIONS = {
    'nothing': nothing,
    'mel_spectrogram_pre_saved': mel_spectrogram_pre_saved,
}
