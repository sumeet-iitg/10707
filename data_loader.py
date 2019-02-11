import numpy as np
import os
from matplotlib import pyplot as plt
import sys

def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)))
    plt.imshow(two_d, interpolation='nearest')
    return plt

def load_instance(line, test=False):
    pixels = line.split(",")
    label = pixels[-1]
    digit1 = []
    digit2 = []
    i=0
    while i < len(pixels) -1:
        digit1.extend(pixels[i:i+28])
        digit2.extend(pixels[i+28:i+56])
        i+=56

    return np.array(digit1, dtype=float), np.array(digit2, dtype=float), label


def load_data(filepath):
    digits = []
    labels = []
    with open(filepath, 'r') as fp:
        filename, _ = os.path.splitext(os.path.basename(filepath))
        for line in fp.readlines():
            one_hot_label = np.zeros(19)
            dig1, dig2, label = load_instance(line)
            one_hot_label[int(label)] = 1
            digits.append(dig1+dig2)
            labels.append(one_hot_label)
            # gen_image(dig1).show()
            # gen_image(dig2).show()
    return digits, labels
            

