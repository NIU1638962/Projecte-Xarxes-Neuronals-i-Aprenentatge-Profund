# -*- coding: utf-8 -*- noqa
"""
Created on Wed Apr 30 03:51:56 2025

@author: Joel Tapia Salvador
"""
import environment
import executions
import utils


def random_split(data, percentage_test: float):

    if environment.SEED is not None:
        environment.torch.manual_seed(environment.SEED)

    masks = {}

    (
        masks[executions.PHASES[0]],
        masks[executions.PHASES[1]],
    ) = environment.torch.utils.data.random_split(
        range(len(data['tracks'])),
        [1 - percentage_test, percentage_test],
    )

    tracks = {
        phase: [
            data['tracks'][index] for index in masks[phase]
        ] for phase in executions.PHASES
    }

    genres = [{
        phase: data['genres'][masks[phase]] for phase in executions.PHASES
    }]

    genres_information = {
        'parents': data['genres_parents'],
        'levels': data['genres_levels'],
        'parents_ids': data['genres_parents_ids'],
        'ids': data['genres_ids'],
    }

    yield tracks, genres, genres_information


def genre_subset_split(
    data,
    percentage_test: float,
    number_genres: int = None,
    selected_genres_indices: list = None,
    min_songs_for_genre: int = None,
    max_tracks: int = None,
    balance_mode: str = 'none',  # 'equal' requiere mismo nº de canciones por género
):
    tracks = data['tracks']
    genres = data['genres']
    genres_parents = data['genres_parents']
    genres_parents_ids = data['genres_parents_ids']
    genres_levels = data['genres_levels']
    genres_ids = data['genres_ids']

    utils.print_message(
        environment.SEPARATOR_LINE
        + '\nSPLITTING BY GENRE SUBSET'
    )

    if environment.SEED is not None:
        environment.torch.manual_seed(environment.SEED)
        environment.numpy.random.seed(environment.SEED)

    if max_tracks is None and environment.LOG_LEVEL_NAME == 'DEBUG':
        max_tracks = 100

    total_genres = genres.shape[1]
    genre_counts = genres.sum(dim=0)

    if selected_genres_indices is not None:
        selected_indices = environment.torch.tensor(selected_genres_indices)
        if min_songs_for_genre is not None:
            insufficient = [
                int(i) for i in selected_indices
                if genre_counts[i] < min_songs_for_genre
            ]
            if insufficient:
                raise ValueError(
                    "Los siguientes géneros no tienen al menos"
                    + f" {min_songs_for_genre} canciones: {insufficient}"
                )
    elif number_genres is not None:
        if min_songs_for_genre is not None:
            eligible_indices = (genre_counts >= min_songs_for_genre).nonzero(
                as_tuple=True)[0]
            if len(eligible_indices) < number_genres:
                raise ValueError(
                    f"Solo hay {len(eligible_indices)} géneros con al menos"
                    + f" {min_songs_for_genre} canciones. "
                    + f"No se pueden seleccionar {number_genres}."
                )
            selected_indices = eligible_indices[environment.torch.randperm(
                len(eligible_indices))[:number_genres]]
        else:
            selected_indices = environment.torch.randperm(total_genres)[
                :number_genres]
    else:
        raise ValueError(
            "You must provide either 'number_genres'"
            + " or 'selected_genres_indices'."
        )

    selected_indices = selected_indices.sort().values

    index_mapping = {
        original_idx: new_idx for new_idx, original_idx in enumerate(
            selected_indices.tolist(),
        )
    }
    genres_parents_filtered = genres_parents[selected_indices]
    genres_levels_filtered = genres_levels[selected_indices]
    genres_parents_ids_filtered = genres_parents_ids[selected_indices]
    genres_ids_filtered = genres_ids[selected_indices]

    genres_parents_remapped = environment.torch.tensor([
        index_mapping.get(int(parent_idx), -1)
        for parent_idx in genres_parents_filtered
    ])

    utils.print_message(f"Selected genre indices: {selected_indices.tolist()}")

    pickle_path = environment.os.path.join(
        environment.PICKLES_PATH,
        (
            f'{environment.LOG_LEVEL_NAME}--{environment.SEED}'
            + f'--{percentage_test}--{number_genres}'
            + f'--{selected_genres_indices}--{min_songs_for_genre}'
            + f'--{max_tracks}--{balance_mode}'
            + '--genre_subset_split--split'
            + '.pickle'
        ),
    )

    if environment.os.path.exists(pickle_path):
        utils.print_message(f'Cargando split cacheado desde: {pickle_path}')
        yield utils.load_pickle(pickle_path)
        return

    genres_subset = genres[:, selected_indices]
    mask_valid_tracks = genres_subset.sum(dim=1) > 0
    tracks_filtered = [t for i, t in enumerate(tracks) if mask_valid_tracks[i]]
    genres_filtered = genres_subset[mask_valid_tracks]

    if max_tracks is not None:
        if balance_mode == 'equal':
            tracks_per_genre = max_tracks // len(selected_indices)
            indices_selected = []
            genres_filtered = genres_filtered.int()

            for i, genre_idx in enumerate(selected_indices):
                mask = genres_filtered[:, i] == 1
                candidates = mask.nonzero(as_tuple=True)[0]

                if len(candidates) < tracks_per_genre:
                    raise ValueError(
                        "No hay suficientes canciones para el género"
                        + f" {int(genre_idx)}: se necesitan {tracks_per_genre}"
                        + f", pero hay {len(candidates)}"
                    )

                perm = environment.torch.randperm(len(candidates))[
                    :tracks_per_genre]
                indices_selected.extend(candidates[perm].tolist())

            indices_selected = environment.torch.tensor(indices_selected)
            tracks_filtered = [tracks_filtered[i] for i in indices_selected]
            genres_filtered = genres_filtered[indices_selected]

        else:
            if len(tracks_filtered) > max_tracks:
                perm = environment.torch.randperm(
                    len(tracks_filtered))[:max_tracks]
                tracks_filtered = [tracks_filtered[i] for i in perm]
                genres_filtered = genres_filtered[perm]

    genres_filtered_numpy = genres_filtered.numpy()

    split_generator = environment.iterstrat.ml_stratifiers.MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=percentage_test,
        random_state=environment.SEED,
    )

    train_idx, val_idx = next(
        split_generator.split(X=genres_filtered_numpy, y=genres_filtered_numpy)
    )

    track_split = {
        'train': [tracks_filtered[i] for i in train_idx],
        'validation': [tracks_filtered[i] for i in val_idx],
    }

    genre_split = {
        'train': genres_filtered[train_idx],
        'validation': genres_filtered[val_idx],
    }

    genres_information = {
        'parents': genres_parents_remapped,
        'levels': genres_levels_filtered,
        'parents_ids': genres_parents_ids_filtered,
        'ids': genres_ids_filtered,
    }

    assert environment.torch.all(
        (
            genre_split['train'].sum(0) + genre_split['validation'].sum(0)
        ) == (
            genres_filtered.sum(0)
        )
    ), 'Split together do not equal full subset.'

    utils.print_message(f"Guardando split generado en: {pickle_path}")
    utils.save_pickle(
        (track_split, (genre_split,), (genres_information,)),
        pickle_path,
    )

    yield track_split, (genre_split,), (genres_information,)


def levels_split(
    data,
    percentage_test: float,
    selected_levels=[],
):
    pickle_path = environment.os.path.join(
        environment.PICKLES_PATH,
        (
            f'{environment.LOG_LEVEL_NAME}--{environment.SEED}'
            + f'--{percentage_test}'
            + f'--{"--".join(str(x) for x in selected_levels)}'
            + '--levels_split--split'
            + '.pickle'
        ),
    )

    if environment.os.path.exists(pickle_path):
        yield utils.load_pickle(pickle_path)
    else:
        if environment.LOG_LEVEL_NAME == 'DEBUG':
            perm = environment.torch.randperm(len(data['tracks']))[:100]
            data['tracks'] = [data['tracks'][i] for i in perm]
            data['genres'] = data['genres'][perm]

        genres_numpy = data['genres'].numpy()

        split_generator = environment.iterstrat.ml_stratifiers.MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=percentage_test,
            random_state=environment.SEED,
        )

        train_idx, val_idx = next(
            split_generator.split(X=genres_numpy, y=genres_numpy)
        )

        tracks_split = {
            'train': [data['tracks'][i] for i in train_idx],
            'validation': [data['tracks'][i] for i in val_idx],
        }

        genres_split = []
        all_levels = environment.torch.unique(data['genres_levels'].int())

        for level in all_levels:
            level_int = int(level.item())
            if selected_levels and level_int not in selected_levels:
                continue  # skip this level

            level_mask = data['genres_levels'].int() == level_int

            genres_split.append({
                'train': data['genres'][train_idx][:, level_mask],
                'validation': data['genres'][val_idx][:, level_mask],
            })

        # Mask for filtering genre-level metadata
        genre_level_mask = environment.torch.tensor(
            [
                (int(l) in selected_levels if selected_levels else True)
                for l in data['genres_levels']
            ],
        )

        genres_information = {
            # 'parents': [p for i, p in enumerate(data['genres_parents']) if genre_level_mask[i]],
            'parents': data['genres_parents'][genre_level_mask],
            'levels': data['genres_levels'][genre_level_mask],
            'parents_ids': data['genres_parents_ids'][genre_level_mask],
            'ids': data['genres_ids'][genre_level_mask],
            # 'parents_ids': [pid for i, pid in enumerate(data['genres_parents_ids']) if genre_level_mask[i]],
            # 'ids': [gid for i, gid in enumerate(data['genres_ids']) if genre_level_mask[i]],
        }

        utils.save_pickle(
            (tracks_split, genres_split, genres_information),
            pickle_path,
        )

        yield tracks_split, genres_split, genres_information


AVAILABLE_DATA_SPLITS = {
    'random_split': random_split,
    'genre_subset_split': genre_subset_split,
    'levels_split': levels_split,
}
