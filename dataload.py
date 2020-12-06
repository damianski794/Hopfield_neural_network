import pandas as pd
import numpy as np
from matplotlib import image, pyplot


from skimage.filters import threshold_mean
from skimage.transform import resize

import os

def preprocessing(img, w=128, h=128):
    # Resize image
    img = resize(img, (w,h), mode='reflect')

    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int

    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

def load(path):
    if path == 'data\\cats':
        df = list()
        for file in os.listdir(path):

            cat_image = image.imread(path + '\\' + file)
            df.append(cat_image[::5,::5,0].reshape(1, -1))
        return np.asarray(df, dtype=np.float32)
    else:
        df = pd.read_csv(path, header=None)
        return df.to_numpy()

