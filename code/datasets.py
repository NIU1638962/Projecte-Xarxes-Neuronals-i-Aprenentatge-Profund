# -*- coding: utf-8 -*- noqa
"""
Created on Wed Apr 30 03:26:28 2025

@author: Joel Tapia Salvador
"""
from typing import List, Union

import environment
import executions
import utils


class DatasetTracksReadOnGet(environment.torch.utils.data.Dataset):

    __slots__ = (
        '__audio_files_paths',
        '__number_frames'
        '__sample_rate',
        '__list_tracks_genres',
    )

    def __init__(
        self,
        audio_files_paths: List[str],
        list_tracks_genres: environment.torch.Tensor,
        genres_information,
        sample_rate: int,
        number_frames: int,
    ):
        for tracks_genres in list_tracks_genres:
            assert len(audio_files_paths) == tracks_genres.shape[0]
            del tracks_genres
            utils.collect_memory()

        self.__audio_files_paths = audio_files_paths
        self.__number_frames = number_frames
        self.__sample_rate = sample_rate
        self.__list_tracks_genres = list_tracks_genres

    def __getitem__(self, index):
        track = utils.normalize_audio(
            utils.load_audio(
                self.__audio_files_paths[index],
                self.__sample_rate,
                self.__number_frames,
            )[0]
        ).unsqueeze(0)

        genres_encoding = [
            tracks_genres[index] for tracks_genres in self.__list_tracks_genres
        ]

        assert not environment.torch.isnan(track).any(), f"{index} nan"

        return track, genres_encoding

    def __len__(self):
        return len(self.__audio_files_paths)

    @property
    def tracks_genres(self):
        return self.__tracks_genres


class DatasetWithMelSpectrogram(environment.torch.utils.data.Dataset):

    __slots__ = (
        '__audio_files_paths',
        '__hop_length'
        '__number_fft_points'
        '__number_frames',
        '__number_mel_channels',
        '__power',
        '__sample_rate',
        '__tracks_genres',
        '__to_decibels',
        '__window_length',
    )

    def __init__(
        self,
        audio_files_paths: List[str],
        tracks_genres: environment.torch.Tensor,
        genres_information,
        sample_rate: int,
        number_frames: int,
        number_fft_points: int,
        hop_length: int,
        window_length: Union[int, None],
        power: float,
        number_mel_channels: int,
        to_decibels: bool,
    ):
        assert len(audio_files_paths) == tracks_genres.shape[0]

        self.__audio_files_paths = audio_files_paths
        self.__tracks_genres = tracks_genres

        self.__number_frames = number_frames
        self.__sample_rate = sample_rate
        self.__number_fft_points = number_fft_points
        self.__hop_length = hop_length
        self.__window_length = window_length
        self.__power = power
        self.__number_mel_channels = number_mel_channels
        self.__to_decibels = to_decibels

    def __getitem__(self, index):
        # Load audio file
        track = utils.normalize_audio(
            utils.load_audio(
                file_path=self.__audio_files_paths[index],
                target_sample_rate=self.__sample_rate,
                number_frames=self.__number_frames,
            )[0]
        )

        # Apply the spectrogram transform
        spectrogram = environment.torch.nan_to_num(
            utils.compute_spectrogram(
                waveform=track,
                sample_rate=self.__sample_rate,
                mel_scale=True,
                n_fft=self.__number_fft_points,
                hop_length=self.__hop_length,
                win_length=self.__window_length,
                power=self.__power,
                n_mels=self.__number_mel_channels,
                to_db=self.__to_decibels
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Get the genre/label for the current track
        genres_encoding = self.__tracks_genres[index]

        assert not environment.torch.isnan(spectrogram).any(), f'{index} nan'

        return spectrogram, genres_encoding

    def __len__(self):
        return len(self.__audio_files_paths)

    def __iter__(self):
        for i in range(len(self)):
            yield self.__audio_files_paths[i], *self[i]

    @property
    def audio_files_paths(self):
        return self.__audio_files_paths

    @property
    def tracks_genres(self):
        return self.__tracks_genres


class DatasetWithSpectogramsPreSaved(environment.torch.utils.data.Dataset):
    __slots__ = (
        '__spectrogram_files_paths'
        '__list_tracks_genres',
    )

    def __init__(
        self,
        spectrogram_files_paths: List[str],
        list_tracks_genres: environment.torch.Tensor,
        genres_information,
    ):
        for tracks_genres in list_tracks_genres:
            assert len(spectrogram_files_paths) == tracks_genres.shape[0]
            del tracks_genres
            utils.collect_memory()

        self.__spectrogram_files_paths = spectrogram_files_paths
        self.__list_tracks_genres = list_tracks_genres

    def __getitem__(self, index):
        spectrogram = utils.load_torch(self.__spectrogram_files_paths[index])

        genres_encoding = [
            tracks_genres[index] for tracks_genres in self.__list_tracks_genres
        ]

        return spectrogram, genres_encoding

    def __len__(self):
        return len(self.__spectrogram_files_paths)

    @property
    def list_tracks_genres(self):
        return self.__list_tracks_genres


AVAILABLE_DATASETS = {
    'dataset_tracks_read_on_get': DatasetTracksReadOnGet,
    'dataset_with_mel_spectrogram': DatasetWithMelSpectrogram,
    'dataset_with_spectograms_pre_saved': DatasetWithSpectogramsPreSaved,
}
