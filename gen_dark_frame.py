from common.helpers import DebugArgParser as ArgumentParser
from os import listdir
from PIL import Image
from os.path import join, basename
import cv2
import numpy as np
from tqdm import tqdm
from shutil import move


def parse_args():
    parser = ArgumentParser(description='Generate a dark frame')
    parser.add_argument('input_dir', type=str,
                        help='The input image')
    parser.add_argument('output_dir', type=str,
                        help='The output image')
    parser.add_argument(
        '--max-images', type=int, default=0,
        help='The maximum number of images to use for the dark frame',
        dest='max_images')
    parser.add_argument(
        '--max-value', type=int, default=255,
        help='The maximum value of the image')
    parser.add_argument(
        '--threshold', type=float, default=0.1,
        help='The percentage of the max value below which we consider a pixel to be dark')
    parser.add_argument(
        '--error-dir', type=str, default='errors', dest='error_dir',
        help='The directory to move images that cannot be read')
    return parser.parse_args()


def is_low_contrast(image, max_value=255, lower_percentile=10,
                    upper_percentile=90, spread_threshold=0.2):
    # Convert image to grayscale if it is not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate the histogram
    histogram = cv2.calcHist([gray], [0], None, [max_value + 1], [0, max_value])

    # Calculate the cumulative distribution function (CDF)
    cdf = histogram.cumsum()
    cdf_normalized = cdf / cdf[-1]  # Normalize to range [0, 1]

    # Calculate the intensity values at the lower and upper percentiles
    lower_value = np.searchsorted(cdf_normalized, lower_percentile / 100.0)
    upper_value = np.searchsorted(cdf_normalized, upper_percentile / 100.0)

    # Calculate the spread
    spread = (upper_value - lower_value) / max_value

    # Determine if the spread is below the threshold
    return spread < spread_threshold


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    max_images = args.max_images
    files = sorted([f for f in listdir(input_dir) if f.endswith('.tif')])
    dark_frame = None
    imgs = 0
    for f in tqdm(files):
        try:
            img = Image.open(join(input_dir, f))
            img = np.array(img)
        except Exception as e:
            print(f'Error reading {f}: {e}')
            f = basename(f)  # Get the filename without the path
            # Move the file to an errors directory
            move(join(input_dir, f), join(args.error_dir, f))
            continue
        if is_low_contrast(img, max_value=args.max_value,
                           spread_threshold=args.threshold):
            if dark_frame is None:
                dark_frame = np.zeros_like(img, dtype=np.float64)
            dark_frame += img
            imgs += 1
        if max_images > 0 and imgs >= max_images:
            break

    dark_frame /= imgs
    dark_frame = np.clip(dark_frame, 0, args.max_value) # Clip the values to the max value
    dark_frame = dark_frame.astype(np.uint8) if args.max_value == 255 else dark_frame.astype(np.uint16)
    cv2.imwrite(join(output_dir, 'dark_fram.tif'), dark_frame)


if __name__ == '__main__':
    main()
