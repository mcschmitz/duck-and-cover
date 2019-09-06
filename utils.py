import os


def create_dir(directory_path):
    """
    Creates a directory to the given path if it does not existö
    Args:
        directory_path: path directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
