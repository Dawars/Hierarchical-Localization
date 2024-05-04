from typing import Tuple
from pathlib import Path
import numpy as np
import cv2
import h5py

def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def list_h5_names(path):
    names = []
    with h5py.File(str(path), 'r', libver='latest') as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip('/'))
        fd.visititems(visit_fn)
    return list(set(names))


def get_keypoints(path: Path, name: str,
                  return_uncertainty: bool = False) -> np.ndarray:
    with h5py.File(str(path), 'r') as hfile:
        dset = hfile[name]
        p = dset.__array__()
    if return_uncertainty:
        return p, None
    return p


def find_pair(hfile: h5py.File, name0: str, name1: str):
    return ' '.join([name0, name1]), False

def get_matches(path: Path, name0: str, name1: str) -> Tuple[np.ndarray]:
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair].__array__().T
        scores = np.ones((matches.shape[0],))
    return matches, scores
