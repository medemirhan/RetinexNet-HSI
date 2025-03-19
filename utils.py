import numpy as np
from PIL import Image
import scipy.io as sio

class Struct:
    pass

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0

def save_images(filepath, result_1, result_2=None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2) if result_2 is not None else None

    if result_2 is None or not np.any(result_2):
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis=1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')

def save_hsi(filepath, data, postfix=None, key='data'):
    """Save hyperspectral image as .mat file"""
    data = np.squeeze(data)
    savepath = filepath[:-4]
    if postfix is not None:
        savepath += postfix
    sio.savemat(savepath + '.mat', {key: data})

def self_normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

def global_normalization(x, max_val, min_val):
    return (x - min_val) / (max_val - min_val + 1e-8)

def per_channel_normalization(x):
    min_val = np.min(x, axis=(0,1), keepdims=True)
    max_val = np.max(x, axis=(0,1), keepdims=True)
    return (x - min_val) / (max_val - min_val + 1e-8)

def per_channel_standardization(x):
    mean = np.mean(x, axis=(0,1), keepdims=True)
    std = np.std(x, axis=(0,1), keepdims=True)
    return (x - mean) / (std + 1e-8)

def load_hsi(file, matContentHeader='data', normalization=None, max_val=None, min_val=None):
    mat = sio.loadmat(file)
    mat = mat[matContentHeader]
    mat = mat.astype('float32')

    x = np.array(mat, dtype='float32')

    if normalization == 'self':
        x = self_normalization(x)
    elif normalization == 'global_normalization':
        x = global_normalization(x, max_val, min_val)
        x[x < 0] = 0.
    elif normalization == 'per_channel_normalization':
        x = per_channel_normalization(x)
    elif normalization == 'per_channel_standardization':
        x = per_channel_standardization(x)
    elif normalization is None:
        pass
    else:
        raise NotImplementedError(normalization + ' is not implemented')

    return (x.astype("float32") / np.max(x))
