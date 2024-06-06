from typing import List
import skimage as ski
import numpy as np
import os
from shutil import copyfile


def is_noisy(file_path):
    img = ski.io.imread(file_path)
    # convert to float
    img = img.astype(np.float32)
    # normalize to the range 0-1
    img = (img - img.min()) / (img.max() - img.min())
    # calculate the standard deviation
    std = img.std()
    return std > 0.05


def main(
        sample_n: int = 10, skip_noisy: bool = True, classes: List[str] = None,
        out_dir: str = None, directory: str = '.', mkdir: bool = True):
    # Sample n files from each class
    # This is a simple function that samples n files from each class in the dataset
    # It is useful for creating a smaller dataset for testing
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]
    if classes is None:
        classes = list(set([f.split('-')[0] for f in files]))

    # filter any noisy files if skip_noisy is True
    if skip_noisy:
        files = [f for f in files if not is_noisy(f)]

    # Split the files into classes
    class_files = {c: [f for f in files if f.startswith(c)] for c in classes}

    # Randomly sample n files from each class
    sampled_files = []
    for c in classes:
        sampled_files.extend(np.random.choice(
            class_files[c], sample_n, replace=False))

    # copy the sampled files to a new directory
    if out_dir is not None:
        if mkdir:
            os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.join(directory, 'sampled')
        if mkdir:
            os.makedirs(out_dir, exist_ok=True)
        for f in sampled_files:
            copyfile(f, os.path.join(out_dir, os.path.basename(f)))

if __name__ == '__main__':
    from common.helpers import DebugArgParser
    parser = DebugArgParser()
    parser.add_argument('--sample_n', type=int, default=10)
    parser.add_argument('--skip_noisy', type=bool, default=True)
    parser.add_argument('--classes', type=str, nargs='+', default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--directory', type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))