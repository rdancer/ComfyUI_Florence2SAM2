import datetime
import os
import shutil
import uuid


def create_directory(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def delete_directory(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")

    try:
        shutil.rmtree(directory_path)
    except PermissionError:
        raise PermissionError(
            f"Permission denied: Unable to delete '{directory_path}'.")


def generate_unique_name():
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4()
    return f"{current_datetime}_{unique_id}"
