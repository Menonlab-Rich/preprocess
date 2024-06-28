from skimage import transform, filters, measure
from PIL import Image
import numpy as np
from glob import glob
from os import path
from uuid import uuid4


def gen_threshold_mask(img, block_size=32):
    def sum_over_blocks(image, block_size):
        for i in range(0, image.shape[0], block_size):
            for j in range(0, image.shape[1], block_size):
                yield np.sum(image[i:i + block_size, j:j + block_size])

    # Apply Gaussian Blur
    cols = img.shape[1]
    rows = img.shape[0]
    # normalize to [0, 1]
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # Sum over blocks
    sums = np.array(list(sum_over_blocks(img, block_size)))
    # reshape the sums to the correct shape
    new_height = int(np.ceil(rows / block_size))
    new_width = int(np.ceil(cols / block_size))
    sums = sums[:new_height * new_width]
    sums = sums.reshape(new_height, new_width)

    # convert to 0 - 1
    sums = (sums - np.min(sums)) / (np.max(sums) - np.min(sums))

    # Threshold the image to keep only the brightest regions
    thresh = filters.threshold_otsu(sums, nbins=(2**16))
    # add a small margin to the threshold (20% of the difference between the max and the threshold)
    # thresh -= 0.2 * (np.max(sums) - thresh)
    binary = sums > thresh

    # Expand the binary image to the original size
    binary = transform.resize(binary, (rows, cols), order=0)

    return binary


def bounding_box(mask):
    x, y, w, h = measure.regionprops(mask.astype(int))[0].bbox
    return x, x + w, y, y + h


def split_image(img, superpixel_size=32):
    # Threshold the image
    mask = gen_threshold_mask(img, block_size=superpixel_size)
    bb = bounding_box(mask)
    img = img[bb[0]:bb[1], bb[2]:bb[3]]

    # Pad the image to make it square and divisible by the superpixel size
    height, width = img.shape
    max_dim = max(height, width)
    pad_height = (superpixel_size - height % superpixel_size) % superpixel_size
    pad_width = (superpixel_size - width % superpixel_size) % superpixel_size

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

    # Split the image into superpixels
    superpixels = []
    for i in range(0, img.shape[0], superpixel_size):
        for j in range(0, img.shape[1], superpixel_size):
            sp = img[i:i + superpixel_size, j:j + superpixel_size]
            if not np.any(sp):
                continue  # Skip empty superpixels
            superpixels.append(sp)
    
    return superpixels


def save_superpixels(superpixels, path, classname='605'):
    # Create a non-overlapping ID for each superpixel to avoid overwriting
    uuid = uuid4()
    for superpixel in superpixels:
        superpixel = Image.fromarray(superpixel)
        superpixel.save(f'{path}/{classname}_{uuid}.tif')


def load_image(path):
    img = Image.open(path)
    img = np.array(img)
    return img


def parse_args():
    from common.helpers import DebugArgParser
    parser = DebugArgParser(description='Split an image into superpixels')
    parser.add_argument(
        'img_pattern', type=str,
        help='Glob pattern to the images to split into superpixels')
    parser.add_argument(
        '--output', type=str, required=True,
        help='Path to the folder where to save the superpixels')
    parser.add_argument(
        '--superpixel-size', type=int, default=32,
        help='Size of the superpixels', dest='superpixel_size')
    parser.add_argument(
        '--create', action='store_true',
        help='Create the output folder if it does not exist')
    return parser.parse_args()


def main():
    from tqdm import tqdm
    import os
    args = parse_args()
    img_paths = glob(args.img_pattern)
    if args.create:
        os.makedirs(args.output, exist_ok=True)
    for img_path in tqdm(img_paths):
        img = load_image(img_path)
        superpixels = split_image(img, superpixel_size=args.superpixel_size)
        save_superpixels(superpixels, args.output,
                         classname=path.basename(img_path)[:3])


if __name__ == '__main__':
    main()
