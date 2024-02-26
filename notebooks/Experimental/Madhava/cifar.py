# stdlib
from pathlib import Path
import pickle
import tarfile
from typing import List
from typing import Optional
from typing import Tuple

# third party
import matplotlib.pyplot as plt
import numpy as np

# syft absolute
from syft import autocache

CIFAR_10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
Batch = Tuple[List[np.ndarray], List[int]]


def unpickle(file: str) -> dict:
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def download_data():
    cifar_tar_gz = autocache(CIFAR_10_URL)
    extract_to_path = './'

    # Open the tar file
    with tarfile.open(cifar_tar_gz, 'r') as tar:
        # Extract all the contents into the directory specified
        tar.extractall(path=extract_to_path)
    file_path = "./cifar-10-batches-py"
    return file_path

def get_batch(filename: str) -> Batch:
    path = download_data()
    batch_data = Path(path + "/" + filename)
    batch_dict = unpickle(batch_data)
    return batch_dict[b"data"], batch_dict[b"labels"]

def get_train_batch(number: int) -> Batch:
    return get_batch(filename=f"data_batch_{number}")

def get_test() -> Batch:
    return get_batch(filename=f"test_batch")

LABEL_MAP = {
    0:"airplane",
    1:"automobile",
    2:"bird",
    3:"cat",
    4:"deer",
    5:"dog",
    6:"frog",
    7:"horse",
    8:"ship",
    9:"truck",
}

def draw(array: np.ndarray, label: Optional[int] = None) -> None:
    image_array_reshaped = array.reshape((3, 32, 32))
    image_array_correct = image_array_reshaped.transpose((1, 2, 0))
    plt.imshow(image_array_correct)
    plt.axis('off')
    plt.show()
    if label:
        print(LABEL_MAP[label])
