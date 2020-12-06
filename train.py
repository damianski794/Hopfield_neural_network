# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt
import skimage.data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
import network
import dataload

# Utils
def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

def reshape(data, width, height):
    # dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (width, height))
    return data

def plot(data, test, predicted, width, height, figsize=(5, 6)):
    data = [reshape(d, width, height) for d in data]
    test = [reshape(d, width, height) for d in test]
    predicted = [reshape(d, width, height) for d in predicted]

    num_fig = 7 if len(data) > 7 else len(data)

    fig, axarr = plt.subplots(num_fig, 3, figsize=figsize)
    for i in range(num_fig):
        if i==0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')

        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

# def preprocessing(img, w=128, h=128):
#     # Resize image
#     img = resize(img, (w,h), mode='reflect')
#     print(img)
#
#     # Thresholding
#     thresh = threshold_mean(img)
#     binary = img > thresh
#     shift = 2*(binary*1)-1 # Boolian to int
#
#     # Reshape
#     flatten = np.reshape(shift, (w*h))
#     return flatten

def main():
    # Load data
    # camera = skimage.data.camera()
    # astronaut = rgb2gray(skimage.data.astronaut())
    # horse = skimage.data.horse()
    # coffee = rgb2gray(skimage.data.coffee())
    animals = dataload.load('data\\animals-14x9.csv')  # h = 9, w = 14
    large25 = dataload.load('data\\large-25x25.csv')  # h = 25, w = 25
    large25plus = dataload.load('data\\large-25x25.plus.csv')  # h = 25, w = 25
    large50 = dataload.load('data\\large-25x50.csv')  # h = 50, w = 25
    letters = dataload.load('data\\letters-14x20.csv')  # h = 20, w = 14
    lettersabc = dataload.load('data\\letters-abc-8x12.csv')  # h = 12, w = 8
    ocra = dataload.load('data\\OCRA-12x30-cut.csv')  # h = 30, w = 12
    small = dataload.load('data\\small-7x7.csv')  # h = 7, w = 7

    # Marge data
    data = animals
    height = 9
    width = 14
    # Preprocessing
    # print("Start to data preprocessing...")
    #data = [preprocessing(d, width, height) for d in data]
    # Create Hopfield Network Model
    model = network.HopfieldNetwork()
    model.train_weights(data)

    # Generate testset
    test = [get_corrupted_input(d, 0.3) for d in data]

    predicted = model.predict(test, threshold=0, asyn=True)
    print("Show prediction results...")
    plot(data, test, predicted, height, width)
    print("Show network weights matrix...")
    #model.plot_weights()

if __name__ == '__main__':
    main()
