import numpy as np
from os import makedirs
from urllib import request
from urllib.error import HTTPError
import gzip
import pickle
import socket

socket.setdefaulttimeout(3)  # timeout in 3 seconds if the connection hangs
folder = "downloads/mnist-pkl/"
filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


def download_mnist():
    # Create the directory to hold the dataset if it doesn't exist
    makedirs(folder, exist_ok=True)
    base_url = "http://yann.lecun.com/exdb/mnist/"

    for name in filename:
        print("Downloading " + name[1] + "...")

        try:
            # Download the files from the URL
            request.urlretrieve(base_url + name[1], folder + name[1])
        except HTTPError:
            # Backup site in case the other one doesn't work
            base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
            request.urlretrieve(base_url + name[1], folder + name[1])

    print("Download complete.")


def save_mnist():
    # Save the images and labels into a pickle file
    mnist = {}

    for name in filename[:2]:
        with gzip.open(folder + name[1], "rb") as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)

    for name in filename[-2:]:
        with gzip.open(folder + name[1], "rb") as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    with open(folder + "mnist.pkl", "wb") as f:
        pickle.dump(mnist, f)

    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load():
    # Load the training and testing data from the pickle file
    with open(folder + "mnist.pkl", "rb") as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], \
        mnist["test_labels"]


if __name__ == "__main__":
    init()
