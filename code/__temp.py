# -*- coding: utf-8 -*- noqa
"""
Created on Fri May 23 03:33:30 2025

@author: JoelT
"""
   if by_levels:
        change_indices = (environment.torch.where(
            genres_levels_tensor[1:] != genres_levels_tensor[:-1],
        )[0] + 1).tolist()

        tracks_genres_tuple = environment.torch.tensor_split(
            tracks_genres_tensor,
            change_indices,
        )

        genres_parents_tuple = environment.torch.tensor_split(
            genres_parents_tensor,
            change_indices,
        )

        genres_levels_tuple = environment.torch.tensor_split(
            genres_levels_tensor,
            change_indices,
        )

        del change_indices
        utils.collect_memory()

    else:
        tracks_genres_tuple = (tracks_genres_tensor,)
        genres_parents_tuple = (genres_parents_tensor,)
        genres_levels_tuple = (genres_levels_tensor,)

    del tracks_genres_tensor, genres_parents_tensor, genres_levels_tensor
    utils.collect_memory()

    data = {
        'tracks': audio_files_paths_list,
        'genres': tracks_genres_tuple,
        'genres_parents': genres_parents_tuple,
        'genres_levels': genres_levels_tuple,
    }
