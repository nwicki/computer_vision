import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

BATCH_SIZE = 16


def distance(x, X):
    return torch.linalg.norm(X - x, dim=1)


def distance_batch(x, X):
    return torch.linalg.norm(X - x[:, None, ...], dim=2)


def gaussian(dist, bandwidth):
    return torch.exp(-0.5 * (dist / bandwidth) ** 2)


def update_point(weight, X):
    return torch.sum(X * weight[:, None], dim=0) / torch.sum(weight)


def update_point_batch(weight, X):
    weighted_sum = torch.sum(X * weight[:, ..., None], dim=1)
    weight_sum = torch.sum(weight, dim=1)
    res = weighted_sum / weight_sum[:, None]
    return res


def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_


def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    frac = int(len(X) / BATCH_SIZE)
    batch_X = X.repeat(BATCH_SIZE, 1, 1)
    for i in range(frac):
        batch = X[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        dist = distance_batch(batch, batch_X)
        weight = gaussian(dist, bandwidth)
        X_[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = update_point_batch(weight, batch_X)
    approx = frac*BATCH_SIZE
    batch_X = X.repeat(len(X) - approx, 1, 1)
    batch = X[approx:]
    dist = distance_batch(batch, batch_X)
    weight = gaussian(dist, bandwidth)
    X_[approx:] = update_point_batch(weight, batch_X)
    return X_


def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X


scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab).cuda()).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
