# -*- coding: utf-8 -*- noqa
"""
Created on Sun Apr 27 02:35:34 2025

@author: Joel Tapia Salvador
"""
from typing import Dict, List, Tuple, Union

import environment
import utils


def check_integrity() -> Dict[str, Tuple[str]]:
    """
    Checks the integrity of all files using the "sha1" checksum of them.

    Returns
    -------
    files_results : Dictionary[Sting, Tuple[Sting]]
        Dictionary with a tuple of all valid files for every subfolder.

    """
    utils.print_message(
        environment.SEPARATOR_LINE
        + '\nCHECKING DATA INTEGRITY'
    )

    results = []
    files_results = {}

    for file in environment.os.scandir(environment.DATA_PATH):
        if file.is_dir():
            subfolder = file.path
            folder_name = environment.os.path.split(subfolder)[1]
            utils.print_message(
                f'{environment.SECTION * 3}'
                + '\nCHECKING FOLDER: '
                + f'"{folder_name}"'
            )

            temp_valid_files = []
            temp_invalid_files = []

            checksums = utils.load_csv(
                environment.os.path.join(subfolder, 'checksums'),
                delimiter='  ',
                header=None,
            )

            for index, (expected_checksum, file_name) in checksums.iterrows():
                file_name = environment.os.path.join(*file_name.split('/'))

                utils.print_message(
                    f'{environment.SEPARATOR * 3}'
                    + f'Checking file: "{file_name}"'
                    + f'{environment.SEPARATOR * 3}'
                )

                file_path = environment.os.path.join(subfolder, file_name)

                if environment.os.path.exists(file_path):

                    obtained_checksum = utils.get_hash_file(
                        'sha1',
                        file_path,
                    )

                    utils.print_message(
                        f'Expected checksum: {obtained_checksum}\n'
                        + f'Obtained checksum: {expected_checksum}'
                    )

                    if expected_checksum == obtained_checksum:
                        utils.print_message('PASSED: CHECKSUMS MATCH')
                        temp_valid_files.append(file_path)

                    else:
                        utils.print_message('ERROR: CHECKSUMS DON NOT MATCH')
                        temp_invalid_files.append(file_path)

                    del obtained_checksum

                else:
                    utils.print_message('ERROR: FILE NOT FOUND')
                    temp_invalid_files.append(file_path)

                del (
                    index,
                    file_name,
                    expected_checksum,
                    file_path,
                )
                utils.collect_memory()

            result = (
                f'For folder "{folder_name}", found {len(temp_valid_files)}'
                + f' valid files out of {checksums.shape[0]} files checksums.'
            )

            results.append(result)

            utils.print_message(
                f'{environment.SEPARATOR * 3}'
                + 'Result'
                + f'{environment.SEPARATOR * 3}\n'
                + result
            )

            files_results[folder_name] = {
                'valids': temp_valid_files,
                'invalids': temp_invalid_files,

            }

            del (
                temp_valid_files,
                temp_invalid_files,
                checksums,
                subfolder,
                folder_name,
                result,
            )
            utils.collect_memory()

    utils.print_message(
        f'{environment.SECTION * 3}'
        + 'FINAL RESULTS:'
        + f'{environment.SECTION * 3}\n'
        + "\n".join(results)
    )

    del results
    utils.collect_memory()

    return files_results


def encode_genres(
    genres: environment.pandas.Series,
    track_genres_information: List[Dict[str, str]],
) -> List[int]:
    """
    Encode genres in a multi-label binary array.

    Parameters
    ----------
    genres :Pandas Series
        Pandas Series with the genres ID ordered.
    track_genres_information : List[Dictionary[String: String]]
        Genres information of a track.

    Returns
    -------
    List[Integers]
        List with binary representing each position a genre an a "0" as
        "no present" or a "1" as "present" in the same order as the genres ID
        of the parameter "genres".

    """
    genres_ids = {
        int(genre['genre_id']): 1 for genre in environment.ast.literal_eval(
            track_genres_information,
        )
    }

    return genres.apply(lambda x: genres_ids.get(x, 0)).to_list()


def get_data() -> Dict[str, Union[List[str], environment.torch.Tensor]]:
    """
    Get the data necessary to create the dataset.

    Returns
    -------
    data : Dictionary[String: List[String] or Torch Tensors]
        Returns a dictionary with the list of file path to the audio files and
        a Torch Tensor with the genres encodings of every track. The indexes
        coindide.

    """
    utils.print_message(
        environment.SECTION_LINE
        + '\nGETTING DATA'
    )

    data_path = environment.os.path.join(
        environment.PICKLES_PATH,
        f'{environment.PLATFORM}--{environment.USER}--data.pickle',
    )

    if not environment.os.path.exists(data_path):

        data_paths_information_path = environment.os.path.join(
            environment.PICKLES_PATH,
            f'{environment.PLATFORM}--{environment.USER}'
            + '--data_paths_information.pickle',
        )

        if not environment.os.path.exists(data_paths_information_path):
            utils.save_pickle(
                check_integrity(),
                data_paths_information_path,
            )

        utils.print_message(
            environment.SEPARATOR_LINE
            + '\nLOADING DATA PATHS'
        )

        data_paths_information = utils.load_pickle(data_paths_information_path)

        del data_paths_information_path
        utils.collect_memory()

        utils.save_pickle(
            data_cleansing(
                data_paths_information=data_paths_information,
            ),
            data_path,
        )

    utils.print_message(
        environment.SEPARATOR_LINE
        + '\nLAODING DATA'
    )

    data = utils.load_pickle(data_path)

    utils.print_message(
        f'Loaded data of "{data["genres"].shape[0]}" tracks.'
    )

    return data


def data_cleansing(
    data_paths_information: Dict[str, Tuple[str]],
) -> Dict[str, Union[List[str], environment.torch.Tensor]]:
    """
    Cleansing of the data.

    Parameters
    ----------
    data_paths_information : Dict[str, Tuple[str]]
        Dictionary with a tuple of all valid files for every subfolder.

    Returns
    -------
    data : Dictionary[String: List[String] or Torch Tensors]
        Returns a dictionary with the list of file path to the audio files and
        a Torch Tensor with the genres encodings of every track. The indexes
        coindide.

    """
    utils.print_message(
        environment.SEPARATOR_LINE
        + '\nDATA CLEANSING'
    )

    utils.print_message(
        f'{environment.SEPARATOR * 3}'
        + 'Reading audio paths'
        + f'{environment.SEPARATOR * 3}'
    )

    audio_files_paths_dictionary = {
        int(
            utils.remove_file_extension(
                environment.os.path.split(file_path)[1],
            )
        ): file_path for file_path in data_paths_information['fma_large']['valids']
    }

    utils.print_message(
        f'{environment.SEPARATOR * 3}'
        + 'Reading metadata paths'
        + f'{environment.SEPARATOR * 3}'
    )

    metadata_files_paths_dictionary = {
        utils.remove_file_extension(
            environment.os.path.split(file_path)[1]
        ): file_path for file_path in data_paths_information['fma_metadata']['valids']
    }

    utils.collect_memory()

    utils.print_message(
        f'{environment.SEPARATOR * 3}'
        + 'Reading genres metadata'
        + f'{environment.SEPARATOR * 3}'
    )

    genres_hierarchy_data_frame = utils.load_csv(
        metadata_files_paths_dictionary['raw_genres'],
    ).set_index(
        'genre_id',
    )[['genre_parent_id']]

    for index, row in genres_hierarchy_data_frame.iterrows():
        parent = genres_hierarchy_data_frame['genre_parent_id'].get(
            row['genre_parent_id'],
        )
        if parent is None or parent == index:
            genres_hierarchy_data_frame.loc[index, 'genre_parent_id'] = -1

        del index, row, parent
        utils.collect_memory()

    for index, row in genres_hierarchy_data_frame.iterrows():
        level = 0.0
        current = row['genre_parent_id']
        while current != -1:
            level += 1
            current = genres_hierarchy_data_frame['genre_parent_id'][current]
        genres_hierarchy_data_frame.loc[index, 'level'] = level

        del index, row, current, level
        utils.collect_memory()

    genres_hierarchy_data_frame = genres_hierarchy_data_frame.sort_values(
        ['level', 'genre_id'],
    )

    genres_series = environment.deepcopy(
        genres_hierarchy_data_frame,
    )[[]].transpose().columns.to_series()

    genres_parents_ids_tensor = environment.torch.tensor(
        environment.deepcopy(
            genres_hierarchy_data_frame
        ).set_index(
            'genre_parent_id',
        )[[]].transpose().columns.to_series().to_list(),
        dtype=environment.torch.float32,
    )

    genres_levels_tensor = environment.torch.tensor(
        environment.deepcopy(
            genres_hierarchy_data_frame
        ).set_index(
            'level',
        )[[]].transpose().columns.to_series().to_list(),
        dtype=environment.torch.float32,
    )

    utils.print_message(
        f'{environment.SEPARATOR * 3}'
        + 'Reading malformed tracks metadata and filtering them'
        + f'{environment.SEPARATOR * 3}'
    )

    malformed_tracks = list(
        utils.load_pickle(
            environment.os.path.join(
                environment.PICKLES_PATH,
                'malformed_tracks.pickle',
            ),
        )['track_id'],
    )

    for key in malformed_tracks:
        audio_files_paths_dictionary.pop(key, None)

    utils.print_message(
        f'{environment.SEPARATOR * 3}'
        + 'Reading tracks metadata and filtering metadata of not found audio files'
        + f'{environment.SEPARATOR * 3}'
    )

    tracks_dataframe = utils.load_csv(
        metadata_files_paths_dictionary['raw_tracks'],
    ).set_index(
        'track_id',
    )[['track_genres']].loc[list(audio_files_paths_dictionary.keys())]

    utils.print_message(
        f'{environment.SEPARATOR * 3}'
        + 'Filtering NAN data'
        + f'{environment.SEPARATOR * 3}'
    )

    for key in tracks_dataframe[tracks_dataframe.isnull().any(axis=1)].index:
        audio_files_paths_dictionary.pop(key, None)

    tracks_dataframe = tracks_dataframe.dropna()

    utils.print_message(
        f'{environment.SEPARATOR * 3}'
        + 'Droping tracks shorter than 29 seconds'
        + f'{environment.SEPARATOR * 3}'
    )

    short_audios_file_path = environment.os.path.join(
        environment.PICKLES_PATH, 'short_tracks.pickle',
    )

    error_audios_file_path = environment.os.path.join(
        environment.PICKLES_PATH, 'error_tracks.pickle',
    )

    if (
            environment.os.path.exists(short_audios_file_path)
    ) and (
            environment.os.path.exists(error_audios_file_path)
    ):
        short_audios_list = utils.load_pickle(short_audios_file_path)
        error_audios_list = utils.load_pickle(error_audios_file_path)

        discarded_audios_list = short_audios_list + error_audios_list

        for key in discarded_audios_list:
            audio_files_paths_dictionary.pop(key, None)
            tracks_dataframe = tracks_dataframe.drop(index=key)

    else:

        short_audios_list = []
        error_audios_list = []

        for key, audio_path in list(audio_files_paths_dictionary.items()):
            try:
                utils.print_message(f'Checking file: {audio_path}')
                if utils.load_audio(audio_path, 1)[0].shape[0] < 29:
                    utils.print_message('Short')
                    short_audios_list.append(key)
                    audio_files_paths_dictionary.pop(key, None)
                    tracks_dataframe = tracks_dataframe.drop(index=key)
                elif utils.load_audio(audio_path, 1)[0].shape[0] >= 29:
                    utils.print_message('Long')
                else:
                    utils.print_message('Error')
                    error_audios_list.append(key)
                    audio_files_paths_dictionary.pop(key, None)
                    tracks_dataframe = tracks_dataframe.drop(index=key)
            except environment.soundfile.LibsndfileError:
                utils.print_message('Error')
                error_audios_list.append(key)
                audio_files_paths_dictionary.pop(key, None)
                tracks_dataframe = tracks_dataframe.drop(index=key)
            except IndexError:
                utils.print_message('Error')
                error_audios_list.append(key)
                audio_files_paths_dictionary.pop(key, None)
                tracks_dataframe = tracks_dataframe.drop(index=key)

        # if audio_files_paths_dictionary.get(118554, False):
        #     raise RuntimeError()

        utils.save_pickle(short_audios_list, short_audios_file_path)
        utils.save_pickle(error_audios_list, error_audios_file_path)

    utils.print_message(
        f'{environment.SEPARATOR * 3}'
        + 'Encoding tracks genres'
        + f'{environment.SEPARATOR * 3}'
    )

    tracks_genres_list = tracks_dataframe['track_genres'].apply(
        lambda x: encode_genres(genres_series, x),
    ).to_list()

    utils.print_message(
        f'{environment.SEPARATOR * 3}'
        + 'Readying data'
        + f'{environment.SEPARATOR * 3}'
    )

    audio_files_paths_list = list(audio_files_paths_dictionary.values())

    tracks_genres_tensor = environment.torch.tensor(
        tracks_genres_list,
        dtype=environment.torch.float32,
    )

    assert len(audio_files_paths_list) == tracks_genres_tensor.shape[0]

    utils.print_message(
        f'Got "{tracks_genres_tensor.shape[0]}" tracks.'
    )

    genres_tensor = environment.torch.tensor(
        genres_series.to_list(),
        dtype=environment.torch.float32,
    )

    genre_id_to_index = {
        genre_id.item(): idx for idx, genre_id in enumerate(genres_tensor)
    }

    genres_parents = environment.torch.tensor(
        [
            genre_id_to_index.get(
                genre_id.item(),
                -1,
            ) for genre_id in genres_parents_ids_tensor
        ],
    )

    data = {
        'tracks': audio_files_paths_list,
        'genres': tracks_genres_tensor,
        'genres_parents_ids': genres_parents_ids_tensor,
        'genres_parents': genres_parents,
        'genres_levels': genres_levels_tensor,
        'genres_ids': genres_tensor,
    }

    return data
