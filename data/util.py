import os
import random
import numpy as np
import glob
import torch
import cv2


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))


def _read_img_lmdb(env, key, size):
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path, size=None):
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(path)
        if size is not None:
            img = cv2.resize(img, (size[0], size[1]))
    else:
        img = _read_img_lmdb(env, path, size)

    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def read_img2(env, path, size=None):
    if env is None:  # img
        img = np.load(path)
        if img is None:
            print(path)
        if size is not None:
            img = cv2.resize(img, (size[0], size[1]))
    else:
        img = _read_img_lmdb(env, path, size)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img



def read_img_seq(path, size=None):
    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))

    img_l = [read_img(None, v, size) for v in img_path_l]
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    try:
        imgs = imgs[:, :, :, [2, 1, 0]]
    except Exception:
        import ipdb; ipdb.set_trace()
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


def read_img_seq2(path, size=None):
    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))

    img_l = [read_img2(None, v, size) for v in img_path_l]
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    try:
        imgs = imgs[:, :, :, [2, 1, 0]]
    except Exception:
        import ipdb; ipdb.set_trace()
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


def index_generation(crt_i, max_n, N, padding='reflection'):
    """Generate an index list for reading N frames from a sequence of images
    Args:
        crt_i (int): current center index
        max_n (int): max number of the sequence of images (calculated from 1)
        N (int): reading N frames
        padding (str): padding mode, one of replicate | reflection | new_info | circle
            Example: crt_i = 0, N = 5
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            new_info: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        return_l (list [int]): a list of indexes
    """
    max_n = max_n - 1
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:
            if padding == 'replicate':
                add_idx = 0
            elif padding == 'reflection':
                add_idx = -i
            elif padding == 'new_info':
                add_idx = (crt_i + n_pad) + (-i)
            elif padding == 'circle':
                add_idx = N + i
            else:
                raise ValueError('Wrong padding mode')
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'new_info':
                add_idx = (crt_i - n_pad) - (i - max_n)
            elif padding == 'circle':
                add_idx = i - N
            else:
                raise ValueError('Wrong padding mode')
        else:
            add_idx = i
        return_l.append(add_idx)
    return return_l


def augment_torch(img_list, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = flip(img, 2)
        if vflip:
            img = flip(img, 1)
        return img

    return [_augment(img) for img in img_list]
