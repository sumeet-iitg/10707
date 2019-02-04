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

    return np.array(digit1,dtype=float), np.array(digit2,dtype=float), label

def load_data(filepath):
    with open(filepath,'r') as fp:
        filename, _ = os.path.splitext(os.path.basename(filepath))
        test = False
        if "test" in filename:
            test = True
        for line in fp.readlines():
            dig1, dig2, label = load_instance(line, test)
            gen_image(dig1).show()
            gen_image(dig2).show()
            exit(0)
            

if __name__ == '__main__':
    directory_path = "C:\\Users\\SumeetSingh\\Documents\\Lectures\\10-707\\HW-Code\\split_data_problem_5_hw1"
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    for file in os.listdir(directory_path):
        load_data(os.path.abspath(os.path.join(directory_path,file)))