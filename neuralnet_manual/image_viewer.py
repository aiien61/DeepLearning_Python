import sys
import os
sys.path.append(os.pardir)
import dataset.load_mnist as lm
import matplotlib.pyplot as plt

dataset = lm.load_mnist()
x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']

# visualise the image to compare the written shape and ideal shape
plt.imshow(dataset['x_test'][8].reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()