"""
Class for managing files. It is mainly a wrapper around existing python functions for ease of use
"""
from pathlib import Path
import os


class PathManager:

    def __init__(self, base_dir_name='FBR', ):
        # base dir
        if base_dir_name != 'scratch':
            self.home = os.path.join(str(Path.home()), base_dir_name)
        else:
            self.home = '/scratch/FBR/'

    def create_dirs(self, relative_path):
        path = self.get_abs_path(relative_path)
        Path(path).mkdir(parents=True, exist_ok=True)

    def get_abs_path(self, relative_path):
        return os.path.join(self.home, relative_path)


path_manager = None


def get_path_manager(base_dir_name='FBR', ):
    return PathManager(base_dir_name=base_dir_name, )
