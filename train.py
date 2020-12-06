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

def check_stability(data, predicted, level):
    good = 0
    average = 0
    data = np.rint(data)
    predicted = np.rint(predicted)
    for i in range(len(data)):
        average += np.count_nonzero(data[i] - predicted[i]) / len(data[i])
        if np.count_nonzero(data[i] - predicted[i]) / len(data[i]) < level:
            good = good+1

    return len(data), good, average/len(data)

def main():
    # Load data
    animals = dataload.load('data\\animals-14x9.csv')  # h = 9, w = 14
    large25 = dataload.load('data\\large-25x25.csv')  # h = 25, w = 25
    large25plus = dataload.load('data\\large-25x25.plus.csv')  # h = 25, w = 25
    large50 = dataload.load('data\\large-25x50.csv')  # h = 50, w = 25
    letters = dataload.load('data\\letters-14x20.csv')  # h = 20, w = 14
    lettersabc = dataload.load('data\\letters-abc-8x12.csv')  # h = 12, w = 8
    ocra = dataload.load('data\\OCRA-12x30-cut.csv')  # h = 30, w = 12
    small = dataload.load('data\\small-7x7.csv')  # h = 7, w = 7

    cats = dataload.load('data\\cats') # h = 300, w = 300

    # Marge data
    data = cats
    height = 300
    width = 300
    # Create Hopfield Network Model
    model = network.HopfieldNetwork()
    model.train_weights(data, 'Hebb')

    # Generate testset
    test = [get_corrupted_input(d, 0.1) for d in data]

    predicted = model.predict(test, threshold=0, asyn=False)
    print("Show prediction results...")
    plot(data, test, predicted, height, width)
    print("Show network weights matrix...")
    # model.plot_weights()
    print(check_stability(data, predicted, 0.1))

if __name__ == '__main__':
    main()