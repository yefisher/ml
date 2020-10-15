import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import imagehash
from PIL import Image
from sklearn import datasets


iris = datasets.load_iris()


def _build():
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['targets'])
    fig, axs = plt.subplots(1, 2)
    n_bins = len(data)
    axs[0].hist(data['sepal length (cm)'], bins=n_bins)
    axs[0].set_title('sepal length')
    axs[1].hist(data['petal length (cm)'], bins=n_bins)
    axs[1].set_title('petal length')
    plt.show()


def get_hash():
    dir_arr = os.listdir('../dataset')
    # todo: create a lib to read each image in each directory and get its hash
    #   already have array of dir where the images are stored
    #   all hashes could be stored in array or smth like that
    #   then get 20% of each 256 hash (last hash's byte should be less then 51)
    print(dir_arr)


if __name__ == "__main__":
    get_hash()
    _build()
