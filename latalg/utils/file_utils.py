from __future__ import annotations
import json

import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List
from distutils.dir_util import copy_tree


def files_with_extension(directory: str, extension: str) -> List[str]:
    """A list of files in a directory with the given extension."""
    files = os.listdir(directory)
    files = [f for f in files if f.endswith(extension)]
    return files


def remove_extension(path: str) -> str:
    """The path or file without the extension."""
    return os.path.splitext(path)[0]


def change_extension(path: str, new_extension: str) -> str:
    """The path or file with a different extension."""
    return remove_extension(path) + '.' + new_extension


def file_name(path: str) -> str:
    """The file name of a path."""
    return Path(path).name


def num_files(path: str) -> int:
    """Number of files in the specified directory."""
    return len(os.listdir(path))


def directory_exists(path: str) -> bool:
    return os.path.isdir(path)


def clear_directory(path: str) -> None:
    """Clear the directory indicated by path, but leave the directory itself."""
    for root, directories, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in directories:
            shutil.rmtree(os.path.join(root, d))


def create_empty_directory(path: str) -> None:
    """Clear a directory if it exists; otherwise, creates a new one."""
    if os.path.exists(path):
        clear_directory(path)
    else:
        os.makedirs(path)


def ensure_created_directory(path: str, clear=False) -> None:
    """If directory does not exist, create it."""
    if clear:
        create_empty_directory(path)
        return

    if not os.path.exists(path):
        os.makedirs(path)


def copy_directory(source_path: str, destination_path: str) -> None:
    """Copy a directory."""
    copy_tree(source_path, destination_path)

def remove_file(path: str) -> None:
    """Remove a file if it exists."""
    if os.path.exists(path):
        os.remove(path)


def read_pickle(path: str) -> Any:
    """Read a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_pickle(path: str, obj: Any) -> None:
    """Write an object to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        

def read_json(path: str) -> Dict:
    """Read a json file."""
    with open(path, 'r') as f:
        return json.load(f)
    

def write_json(path: str, obj: Dict) -> None:
    """Write an object to a json file."""
    with open(path, 'w') as f:
        json.dump(obj, f)


def read_file(path: str) -> str:
    """Read a text file."""
    with open(path, 'r') as file:
        return file.read()


def write_file(path: str, data: str) -> None:
    """Write text to a file."""
    with open(path, 'w') as file:
        file.write(data)