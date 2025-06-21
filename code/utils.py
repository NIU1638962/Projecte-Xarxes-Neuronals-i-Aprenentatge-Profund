# -*- coding: utf-8 -*- noqa
"""
Created on Tue Mar 25 21:29:37 2025

@author: Joel Tapia Salvador
"""
from typing import Tuple, Union

import environment


def compute_weights(list_label_tensor, method="log", epsilon=1e-6):
    list_weights = []
    for label_tensor in list_label_tensor:
        label_tensor = label_tensor.to(environment.TORCH_DEVICE)
        number_samples, number_classes = label_tensor.shape
        positive_count = label_tensor.sum(dim=0).float()
        negative_count = number_samples - positive_count

        if method == 'ratio':
            weights = environment.deepcopy(positive_count)
        elif method == "inver_ratio":
            weights = negative_count / (positive_count + epsilon)
        elif method == "inver_log":
            weights = environment.torch.log(
                (number_samples + 1) / (positive_count + 1)
            )
        elif method == 'none':
            weights = environment.torch.ones(
                number_classes,
                dtype=environment.torch.float32,
            )
        else:
            raise ValueError(
                "Unsupported method: choose 'pos', 'ratio', 'log' or 'none'."
            )

        assert weights.shape[0] == number_classes, 'Not equal amount of weight to classes'

        weights = weights.to(environment.TORCH_DEVICE)

        list_weights.append(weights)

    return list_weights


def compute_spectrogram(
    waveform: environment.torch.Tensor,
    sample_rate: int,
    mel_scale: bool = True,
    n_fft: int = 1024,
    hop_length: int = 512,
    win_length: Union[int, None] = None,
    power: float = 2.0,
    n_mels: int = 128,
    to_db: bool = True,
) -> environment.torch.Tensor:
    waveform = waveform.to(environment.TORCH_DEVICE)
    # Ensure the waveform is in the shape (channels, time)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Select the appropriate transform
    if mel_scale:
        transform = environment.torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length or n_fft,
            power=power,
            n_mels=n_mels,
        ).to(environment.TORCH_DEVICE)
    else:
        transform = environment.torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length or n_fft,
            power=power,
        ).to(environment.TORCH_DEVICE)

    # Apply the chosen transformation (MelSpectrogram or Spectrogram)
    spectrogram = transform(waveform)  # Shape: (1, n_mels/freq_bins, frames)

    # Convert to decibels (logarithmic scale) if needed
    if to_db:
        spectrogram = environment.torchaudio.transforms.AmplitudeToDB(
            top_db=80.0
        )(
            spectrogram,
        )

    # Return (n_mels, frames) or (freq_bins, frames)
    return spectrogram.squeeze(0).cpu()


def collect_memory():
    """
    Collect garbage's collector's' memory and CUDA's cache and shared memory.

    Returns
    -------
    None.

    """
    environment.gc.collect()

    if environment.CUDA_AVAILABLE and environment.TORCH_DEVICE.type == 'cuda':
        environment.torch.cuda.ipc_collect()
        environment.torch.cuda.empty_cache()


def formated_traceback_stack() -> str:
    """
    Get Formated Traceback Stack.

    Returns the Formated Traceback Stack untill the point before calling this
    funtion and formats it to the style of the logs.

    Returns
    -------
    string
        Formated Traceback Stack.

    """
    return 'Traceback (most recent call last):' + (
        '\n' + ''.join(environment.traceback.format_stack()[:-2])[:-1]
    )


def get_disk_space(path: str) -> Tuple[int, int, int]:
    """
    Get information about the path's disk's space.

    Print the disk's space information (total disk space, free disk space,
    used disk space) of the give path in human redable text and return such
    information in bytes.

    Parameters
    ----------
    path : String
        Path to look up disk's space.

    Returns
    -------
    total_disk_space : Integer
        Number of bytes of the total disk's space of the path.
    used_disk_space : Integer
        Number of bytes currently used of the path.
    free_disk_space : Integer
        Number of bytes currently free of the path.

    """
    disk_space = environment.psutil.disk_usage(path)

    total_disk_space = disk_space.total
    free_disk_space = disk_space.free

    used_disk_space = total_disk_space - free_disk_space

    redable_free_disk_space = transform_redable_byte_scale(free_disk_space)
    redable_total_disk_space = transform_redable_byte_scale(total_disk_space)
    redable_used_disk_space = transform_redable_byte_scale(used_disk_space)

    verbose_redable_disk_space_info = (
        f'Total Path Disk Space: {redable_total_disk_space}'
        + f'\nUsed Path Disk Space: {redable_used_disk_space}'
        + f'\nFree Path Disk Space: {redable_free_disk_space}'
    )

    environment.logging.info(
        verbose_redable_disk_space_info.replace('\n', '\n\t\t'))

    print_message(verbose_redable_disk_space_info)

    return total_disk_space, used_disk_space, free_disk_space


def get_hash_file(hash_type: str, file_path: str):
    hash_object = environment.hashlib.new(hash_type)
    with open(file_path, 'rb') as file:
        block = file.read(2**16)
        while len(block) != 0:
            hash_object.update(block)
            block = file.read(2**16)
        return hash_object.hexdigest()


def get_memory_cuda() -> Tuple[int, int, int]:
    """
    Get information about CUDA's device memory.

    Print the memory information (total memory, free memory, used memory) of
    the CUDA device in human redable text and return such information in bytes.

    Returns
    -------
    total_memory : Integer
        Number of bytes of the total memory of the CUDA device.
    used_memory : Integer
        Number of bytes currently used of the CUDA device.
    free_memory : Integer
        Number of bytes currently free of the CUDA device.

    """
    total_memory = 0
    free_memory = 0

    if environment.CUDA_AVAILABLE and environment.TORCH_DEVICE.type == 'cuda':
        free_memory, total_memory = environment.torch.cuda.mem_get_info(
            environment.TORCH_DEVICE
        )

    used_memory = total_memory - free_memory

    redable_free_memory = transform_redable_byte_scale(free_memory)
    redable_total_memory = transform_redable_byte_scale(total_memory)
    redable_used_memory = transform_redable_byte_scale(used_memory)

    verbose_redable_memory_info = (
        f'Total CUDA Memory: {redable_total_memory}'
        + f'\nUsed CUDA Memory: {redable_used_memory}'
        + f'\nFree CUDA Memory: {redable_free_memory}'
    )

    environment.logging.info(
        verbose_redable_memory_info.replace('\n', '\n\t\t'),
    )

    print_message(verbose_redable_memory_info)

    return total_memory, used_memory, free_memory


def get_memory_object(an_object: object) -> int:
    """
    Get object size in bytes.

    Warning: this does not include size of referenced objects inside the
    objejct and is teh result of calling a method of the object that can be
    overwritten. Be careful when using and interpreting results.

    Parameters
    ----------
    an_object : Object
        Object to get the size of.

    Returns
    -------
    size : Integer
        Size in bytes of the object.

    """
    size = environment.sys.getsizeof(an_object)

    return size


def get_memory_system() -> Tuple[int, int, int]:
    """
    Get information about system's memory.

    Print the memory information (total memory, free memory, used memory) of
    the system in human redable text and return such information in bytes.

    Returns
    -------
    total_memory : Integer
        Number of bytes of the total memory of the system.
    used_memory : Integer
        Number of bytes currently used of the system.
    free_memory : Integer
        Number of bytes currently free of the system.

    """
    memory = environment.psutil.virtual_memory()

    total_memory = memory.total
    free_memory = memory.available

    used_memory = total_memory - free_memory

    redable_free_memory = transform_redable_byte_scale(free_memory)
    redable_total_memory = transform_redable_byte_scale(total_memory)
    redable_used_memory = transform_redable_byte_scale(used_memory)

    verbose_redable_memory_info = (
        f'Total System Memory: {redable_total_memory}'
        + f'\nUsed System Memory: {redable_used_memory}'
        + f'\nFree System Memory: {redable_free_memory}'
    )

    environment.logging.info(
        verbose_redable_memory_info.replace('\n', '\n\t\t'),
    )

    print_message(verbose_redable_memory_info)

    return total_memory, used_memory, free_memory


def load_audio(file_path, target_sample_rate=16000, number_frames=-1):
    """
    Load an audio file, convert it to mono if necessary, and resample it to the target sample rate.
    Args:
        filepath (str): Path to the audio file to be loaded.
        target_sr (int, optional): Target sample rate in Hz. Defaults to 16000.
    Returns:
        tuple: A tuple containing:
            - waveform (torch.Tensor): The audio waveform as a 1D tensor.
            - target_sr (int): The sample rate of the returned waveform.
    """
    try:
        waveform, sample_rate = environment.torchaudio.load(
            file_path,
            format='mp3',
        )
    except Exception as e:
        print(f"[ERROR] Could not load {file_path}: {e}")
        return None

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = environment.torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform.squeeze()[:number_frames], target_sample_rate


def load_csv(
    file_path: str,
    encoding: str = environment.locale.getpreferredencoding(),
    delimiter=None,
    header='infer',
    skip_blank_lines: bool = True,
    low_memory: bool = True,
) -> environment.pandas.DataFrame:
    """
    Load and read a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : String
        String with the relative or absolute path of the CSV file.
    encoding : String, optional
        String with the encoding used on the CSV file. The default is
        locale.getpreferredencoding().

    Returns
    -------
    data : Pandas DataFrame
        Pandas Dataframe containing the CSV data.

    """
    if delimiter is None:
        with open(file_path, mode='r', encoding=encoding) as csv_file:
            dialect = environment.csv.Sniffer().sniff(csv_file.readline())

        delimiter = dialect.delimiter

        del csv_file, dialect
        collect_memory()

    data = environment.pandas.read_csv(
        file_path,
        sep=delimiter,
        header=header,
        skip_blank_lines=skip_blank_lines,
        low_memory=low_memory,
    )

    return data


def load_json(
    file_path: str,
    encoding: str = environment.locale.getpreferredencoding(),
) -> object:
    """
    Load a JSON file into a Python Object.

    Parameters
    ----------
    file_path : String
        String with the relative or absolute path of the JSON file.
    encoding : String, optional
        String with the encoding used on the JSON file. The default is
        locale.getpreferredencoding().

    Returns
    -------
    python_object : Object
        Python Object containing the JSON data.

    """
    with open(file_path, mode='r', encoding=encoding) as file:
        python_object = environment.json.load(file)

    del file
    collect_memory()

    return python_object


def load_pickle(file_path: str) -> object:
    """
    Read a binary dump made by pickle of a Python Object.

    Parameters
    ----------
    file_path : Sting
        String with the relative or absolute path of the binary file.

    Returns
    -------
    python_object : Object
        Python Object read from the binary dump made by pickle.

    """
    with open(file_path, mode='rb') as file:
        python_object = environment.pickle.load(file)

    del file
    collect_memory()

    return python_object


def load_torch(file_path, map_location='cpu'):
    return environment.torch.load(
        file_path,
        map_location=map_location,
        weights_only=True,
    )


def module_from_file(
    module_name: str,
    file_path: str,
) -> environment.types.ModuleType:
    """
    Load a Python ModuleType from a file.

    Parameters
    ----------
    module_name : String
        Name of the module.
    file_path : String
        Path to the module file.

    Returns
    -------
    module : Python ModuleType
        Python ModuleType.

    """
    spec = environment.importlib.util.spec_from_file_location(
        module_name, file_path
    )
    module = environment.importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_audio(waveform):
    return waveform / waveform.abs().max()


def print_error(error: Union[str, None] = None):
    stack = formated_traceback_stack()

    if error is None or error == '':
        message = stack
    else:
        message = error + '\n' + stack
    del error, stack

    environment.logging.error(message.replace('\n  ', '\n\t\t'), stacklevel=3)
    print(message)
    del message


def print_message(message: str = ''):
    """
    Print a message to sys.stdout and log it and "info" level.

    Parameters
    ----------
    message : String
        Message to print.

    Returns
    -------
    None.

    """
    environment.logging.info(message.replace('\n', '\n\t\t'), stacklevel=3)
    print(message)
    del message


file_name_without_extension_regex = environment.re.compile(
    r"^(.*)(?=\.[a-zA-Z0-9]+$)"
)


def remove_file_extension(file_path: str) -> str:
    result = file_name_without_extension_regex.search(file_path)

    if result is None:
        return file_path

    return result.group(1)


def safe_division(
        numerator: Union[int, float],
        denominator: Union[int,  float],
) -> float:
    """
    Performs a safe division.

    If denominator is 0 return 0 instead of raising ZeroDivisionError.

    Parameters
    ----------
    numerator : Integer or Float
        DESCRIPTION.
    den : Integer or Float
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    """
    return float(numerator / denominator) if denominator else 0.0


def save_csv(dataframe: environment.pandas.DataFrame, file_path: str):
    """
    Save Pandas DataFrame into a CSV file.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        Pandas DataFrame containing the data to save in the CSV file.
    file_path : str
        Absolute or relative path of the CSV file to save into.

    Returns
    -------
    None.

    """
    dataframe.to_csv(file_path, index=False)


def save_pickle(python_object: object, file_path: str):
    """
    Write a binary dump made by pickle of a Python Object.

    Parameters
    ----------
    python_object : Object
        Python Object to write int a binary dump made by pickle.
    file_path : String
        String with the relative or absolute path of the binary file.

    Returns
    -------
    None.

    """
    with open(file_path, mode='wb') as file:
        environment.pickle.dump(python_object, file)

    del file
    collect_memory()


def save_torch(torch_object, file_path):
    environment.torch.save(torch_object, file_path)


def transform_redable_byte_scale(number_bytes: int) -> str:
    """
    Tranform a number of bytes into the apropiate unit of the scale to read it.

    Parameters
    ----------
    number_bytes : Integer
        Number of bytes to be transformed.

    Returns
    -------
    String
        Numebr of bytes in the most human redable unit of the scale.

    """
    scale_bytes = ('B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB')
    i = 0
    while number_bytes >= 2 ** 10:
        number_bytes = number_bytes / (2 ** 10)
        i += 1

    return f'{number_bytes} {scale_bytes[i]}'
