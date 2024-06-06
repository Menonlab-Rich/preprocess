import os
from tqdm import tqdm
import glob
from PIL import Image, ImageSequence
import numpy as np
import re
import cv2
import multiprocessing as mp
from multiprocessing.managers import ValueProxy
from typing import List, Union, Tuple
from common.func_helpers import flatten
from common.mp_helpers import wait_for_memory
from skimage import restoration
from concurrent.futures import ThreadPoolExecutor


def print(*args):
    '''
    Overwrite the print function to use tqdm.write for better output handling.
    '''
    tqdm.write(" ".join(map(str, args)) + "\n")


def do_normalize(image, darkframe_path, noiseframe_path=None, dtype=None,
                 denoise=False) -> np.ndarray:
    dtype = image.dtype if dtype is None else dtype
    maxval = np.iinfo(dtype).max
    if darkframe_path is not None:
        darkframe = Image.open(darkframe_path)
        darkframe = np.array(darkframe).astype(np.float32)
        darkframe = darkframe / darkframe.max() if darkframe.max() > 0 else darkframe
        darkframe = darkframe * maxval  # Convert to 16-bit range
        image = image / image.max() if image.max() > 0 else image
        image = image * maxval  # Convert to 16-bit range
        image = (image - darkframe)  # Subtract the dark frame
        image = np.clip(image, 0, maxval)  # Clip the values to the 16-bit range
    # Normalize the image
    image = image / image.max() if image.max() > 0 else image
    if denoise:
        # Convert to 8-bit range for denoising
        image = (image * 255).astype(np.uint8)
        image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        image = (image / 255).astype(np.float32)  # Convert back to 0-1 range
        if noiseframe_path is not None:
            noise_frame = Image.open(noiseframe_path)
            noise_frame = np.array(noise_frame).astype(np.float32) / maxval
            image = image - noise_frame
            image = np.clip(image, 0, maxval)

    # Convert back to correct dtype and range
    image = (image * maxval).astype(dtype)

    return image


def expand_brace_pattern(pattern):
    """
    Expand a glob pattern containing curly braces into separate patterns.

    :param pattern: A glob pattern possibly containing a part in curly braces.
    :return: A list of expanded glob patterns.
    """
    # Regex to find the content within curly braces
    brace_pattern = r'\{(.*?)\}'
    match = re.search(brace_pattern, pattern)

    if not match:
        # Return the original pattern in a list if no braces found
        return [pattern]

    # Extract the matched part and split by comma
    parts = match.group(1).split(',')

    # Construct new patterns by replacing the curly brace part with each split part
    base_pattern = pattern.replace(
        match.group(0),
        '{}')  # Placeholder for format
    expanded_patterns = [base_pattern.format(part) for part in parts]

    return expanded_patterns


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

def is_noisy(image, threshold=-18):
    if image.max() > 0:
        image = (image - image.min()) / (image.max() - image.min()) # Normalize to 0-1 range if not already
    
    mean = np.mean(image)
    std = np.std(image)
    snr = 10 * np.log10(mean / std)
    return snr < threshold


def is_emptyish(image, threshold=0.02):
    '''
    Returns True if the image is mostly empty (below the threshold).
    '''
    image = image / image.max() if image.max() > 0 else image
    rms = np.sqrt(np.mean(image**2))
    return rms < threshold


def generate_dark_frame(
        darkframe_path: str, paths: List[str],
        threshold: float = 0.003, max_images: Union[int, None] = None) -> bool:
    dark_frame = None
    imgs = 0
    np_page = None
    for tiff_path in tqdm(paths, desc="Generating Dark Frame"):
        with Image.open(tiff_path) as img:
            for i, page in enumerate(ImageSequence.Iterator(img)):
                mode = page.mode
                if mode != 'I;16' and mode != 'I;16B':
                    print(
                        f"Warning: Image mode for {tiff_path} is '{page.mode}', which may not be correct for 16-bit data. Attempting to adjust.")
                    page = page.convert('I;16')
                np_page = np.array(page)
                page = np.array(
                    do_normalize(
                        np_page, None, None, denoise=True),
                    dtype=np.uint16)
                # check the RMS value of the image to determine if it is page that should be used for dark frame
                if is_low_contrast(
                        page, spread_threshold=threshold, max_value=65535):
                    if dark_frame is None:
                        dark_frame = np.zeros_like(
                            np.array(page, dtype=np.float64),
                            dtype=np.float64)
                    dark_frame += page
                    imgs += 1
                if max_images is not None and imgs >= max_images:
                    break
    if imgs == 0:
        return False

    dark_frame /= imgs
    Image.fromarray(
        do_normalize(
            dark_frame, None, None, dtype=np.uint16, denoise=False)).save(
        darkframe_path)
    return True


def extract_multipage_tiff(input_glob: str, output_dir: str, lock: mp.Lock,
                           critical_event: mp.Event,
                           darkframe_path: ValueProxy[str],
                           create_dir: bool = False, normalize: bool = False,
                           crop: Union[Tuple[int],
                                       int, None] = None,
                           noiseframe_path: Union[str, None] = None,
                           df_threshold: float = 0.008, max_images=100):
    # Critical Section
    input_glob = expand_brace_pattern(input_glob)
    paths = flatten([glob.glob(input_pattern) for input_pattern in input_glob])
    futures = []
    if not critical_event.is_set():
        with lock:
            # Check a second time in case any process is waiting on the lock while another process has already set the event
            if not critical_event.is_set():
                if create_dir:
                    os.makedirs(output_dir, exist_ok=True)

                if darkframe_path.get() is not None:
                    # check if a dark frame exists
                    print("Checking for dark frame.")
                    if not os.path.exists(darkframe_path.get()):
                        print("Generating dark frame.")
                        if (generate_dark_frame(darkframe_path.get(), paths, max_images=max_images, threshold=df_threshold)):
                            print("Dark frame generated.")
                        else:
                            print("No images found for dark frame.")
                            darkframe_path.set(None)
                    else:
                        print("Dark frame found.")
            critical_event.set()
    # End Critical Section

    # Lock is captured here to ensure that each process waits for the previous process to load images before attempting to load more
    with lock:
        # Wait for 2GB of memory to be available
        wait_for_memory(threshold_mb=2048, check_interval=10)
        images = [Image.open(path) for path in paths]
        image_with_path = zip(images, paths)
    for img, path in tqdm(image_with_path, total=len(paths)):
        try:
            base_filename = os.path.basename(path)
            name, _ = os.path.splitext(base_filename)

            for i, page in enumerate(ImageSequence.Iterator(img)):
                mode = page.mode
                output_filename = f"{name}_page_{i+1}.tif"
                output_path = os.path.join(output_dir, output_filename)
                if crop:
                    if len(crop) == 2:
                        # crop represents desired dimensions process accordingly
                        expected_width = crop[0]
                        expected_height = crop[1]
                        left = (page.width - expected_width) // 2
                        top = (page.height - expected_height) // 2
                        right = (page.width + expected_width) // 2
                        bottom = (page.height + expected_height) // 2
                        roi = (left, top, right, bottom)
                    else:
                        left, top, right, bottom = crop

                        roi = (
                            left,
                            top,
                            right,
                            bottom)
                    page = page.crop(roi)
                if normalize:
                    # Ensure page is in the correct mode for 16-bit data if expected
                    if mode != 'I;16' and mode != 'I;16B':
                        print(
                            f"Warning: Image mode for {output_filename} is '{page.mode}', which may not be correct for 16-bit data. Attempting to adjust.")
                        # Attempt to convert to a 16-bit mode; this might not be correct for all images
                        page = page.convert('I;16')

                    np_page = np.array(page)
                    # Check if np_page has the expected range for 16-bit data
                    try:
                        image = np.array(
                            do_normalize(
                                np_page, darkframe_path.get(),
                                noiseframe_path),
                            dtype=np.uint16)
                        if not is_low_contrast(image, spread_threshold=0.006, max_value=65535) and not is_emptyish(image) and not is_noisy(image):
                            Image.fromarray(image).save(output_path)
                    except Exception as e:
                        print(f"Error processing file {path}: {e}")

        except IOError as e:
            print(f"Error processing file {path}: {e}")

    for future in tqdm(futures, desc="Saving Images"):
        img = future.result()
        Image.fromarray(img).save(output_path)


def verify_extracted_outputs(output_dir):
    """
    Verify the extracted images in the output directory.
    """
    files = glob.glob(os.path.join(output_dir, "*.tif"))
    for file in tqdm(files):
        try:
            img = Image.open(file)
            img.verify()
        except Exception as e:
            print(f"Error verifying {file}: {e}")
            os.remove(file)


if __name__ == "__main__":
    from common.helpers import DebugArgParser
    # Set up the argument parser
    parser = DebugArgParser(
        description="Extract pages from multipage TIFF files.")
    parser.add_argument("input", type=str, nargs="+",
                        help="Glob pattern to match TIFF files.")
    parser.add_argument(
        "--output", type=str,
        required=True,
        help="Directory where extracted pages will be saved.")
    parser.add_argument(
        "--normalize", action="store_true",
        help="Normalize the extracted images.")
    parser.add_argument(
        "--create_dir", action="store_true",
        help="Create the output directory if it doesn't exist.")
    parser.add_argument(
        "--crop", nargs="+", type=int,
        help="Remove the specified number of pixels from the edges. (left, upper, right, lower).",
        default=None)
    parser.add_argument(
        "--darkframe", type=str,
        help="Path to the dark frame image for normalization.", default=None)
    parser.add_argument(
        "--df-max-images", type=int, default=100,
        help="Maximum number of images to use for dark frame generation.",
        dest="max_images")
    parser.add_argument(
        "--noiseframe", type=str,
        help="Path to the noise frame image for normalization.", default=None)
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify the extracted images."
    )
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Suppress the confirmation prompt.", dest="no_confirm")
    parser.add_argument(
        '--df-threshold', type=float, default=0.008,
        help="Threshold for dark frame generation.", dest="df_threshold")

    args = parser.parse_args()
    if not args.no_confirm:
        print("This script will extract pages from multipage TIFF files.")
        print("It may take a long time to run.")
        print("Do you want to continue? (y/n)")
        response = input()
        if response.lower() != "y":
            print("Exiting...")
            exit(0)
    pool = mp.Pool(mp.cpu_count())

    if args.crop and (len(args.crop) != 2 and len(args.crop) != 4):
        raise ValueError("Crop argument must have 2, or 4 values.")

    manager = mp.Manager()
    lock = manager.Lock()
    event = manager.Event()
    df_path = manager.Value(str, args.darkframe)
    
    if args.max_images < 0:
        raise ValueError("max_images must be greater than or equal to 0.")
    
    # Set max_images to None if 0 is passed because None can't be passed via the command line
    # But is expected to an int or None in the function
    args.max_images = None if args.max_images == 0 else args.max_images

    futures = [pool.apply_async(
        extract_multipage_tiff,
        (input_pattern, args.output, lock, event, df_path),
        dict(
            normalize=args.normalize, create_dir=args.create_dir,
            crop=args.crop, noiseframe_path=args.noiseframe,
            df_threshold=args.df_threshold, max_images=args.max_images))
        for input_pattern in args.input]
    for future in tqdm(futures):
        future.get()  # Wait for the process to finish

    if args.verify:
        print("Verifying the output images.")
        verify_extracted_outputs(args.output)
        print("Verification complete.")
