import shutil
from pathlib import Path


def clear_output_directory(dir_path='runs'):
    dirpath = Path(dir_path)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
