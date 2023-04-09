import sys, os
# Get the absolute path of the src directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Add the src directory to sys.path
sys.path.append(src_path)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)