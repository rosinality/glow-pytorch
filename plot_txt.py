import sys
import numpy as np
from PIL import Image


def plot_txt(img):
    img = np.array(img)
    img = img.mean(axis=2)
    string = ""
    for i in range(img.shape[0]):
        for j in range(min(img.shape[1], 200)):
            if img[i,j] >= 128:
                string += "O"
            else:
                string += " "
        string += "\n"
    print(string)


f_name = sys.argv[1]
img = Image.open(f_name)
plot_txt(img)
