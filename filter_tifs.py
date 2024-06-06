import os
import cv2
import numpy as np
from shutil import move
from tqdm import tqdm
from skimage import restoration, util, measure, filters
from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import disk
from glob import glob
import concurrent.futures
from imagehash import average_hash
from multiprocessing import Manager, Process
from PIL import Image
import psutil
from time import sleep

def print(*args):
    tqdm.write(" ".join(map(str, args)) + "\n")

class FilterDuplicateImages:
    def __init__(self):
        self.image_hashes = set()

    def __call__(self, image: np.ndarray) -> bool:
        image_hash = average_hash(image)
        if image_hash in self.image_hashes:
            return True
        self.image_hashes.add(image_hash)
        return False

def wait_for_memory(threshold_mb=2048, check_interval=1):
    """
    Pause the current thread until available memory is above the threshold.
    
    :param threshold_mb: Minimum available memory in MB to continue execution.
    :param check_interval: Interval in seconds to check memory availability.
    """
    print("Waiting for available memory...")
    while True:
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        if available_memory_mb >= threshold_mb:
            break
        sleep(check_interval)
    print("Memory available.")

def calculate_image_metrics(image):
    # assume the image is a 2D numpy array normalize from 0 to 1
    # calculate the mean, std, snr, entropy, and contrast
    image = util.img_as_ubyte(image)
    mean = np.mean(image)
    std = np.std(image)
    # Calculate the signal-to-noise ratio
    snr = 10 * np.log10(mean / std) if std > 0 else 0
    # Calculate the entropy in a 5x5 window
    entropy = sk_entropy(image, disk(5))
    entropy = np.mean(entropy)  # Average the entropy values
    contrast = np.max(image) - np.min(image)  # Calculate the contrast
    return mean, std, snr, entropy, contrast

def calculate_rms_contrast(image):
    """
    Calculate the Root Mean Square (RMS) contrast of the image.
    """
    image = util.img_as_float(image)
    mean = np.mean(image)
    rms_contrast = np.sqrt(np.mean((image - mean) ** 2))
    return rms_contrast


def is_low_contrast(image, threshold=0.01):
    """
    Determine if the image is low contrast based on RMS contrast.
    """
    rms_contrast = calculate_rms_contrast(image)
    return rms_contrast < threshold

def snr(image):
    '''
    Calculate the signal-to-noise ratio of the image.
    '''
    normed = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
    mean, std = cv2.meanStdDev(normed)
    snr = 10 * np.log10(mean[0] / std[0])
    return snr


def connected_area(image: np.ndarray):
    '''
    Returns the area of the largest connected component in the binary image.
    '''
    # Convert the image to binary using Otsu's thresholding
    threshold = filters.threshold_otsu(image)
    thresholded = image > threshold
    labels = measure.label(thresholded, connectivity=2, background=0)
    areas = [r.area for r in measure.regionprops(labels)]
    low_areas = [area for area in areas if area < 100]
    high_areas = [area for area in areas if area >= 100]
    return (max(areas), low_areas, high_areas) if areas else (0, [], [])


def is_overexposed(image, threshold=0.95, max_intensity=255):
    # Assume the image is a 2D numpy array normalize from 0 to 1
    image = image[image > 0] # Remove zero values because they only serve to skew the histogram
    image = np.array(image, dtype=np.uint8) * max_intensity
    if image.size == 0:
        return False # Return False if the image is empty
    hist = cv2.calcHist([image], [0], None, [max_intensity + 1], [0, max_intensity + 1])
    # divide the hist into 5 bins (blacks, shadows, mids, highlights, whites)
    hist = hist / hist.sum() # Normalize the histogram
    bins = np.array_split(hist, 5)
    # calculate the percentage of pixels in the last (white) bin
    last_bin = bins[-1]
    last_bin_percentage = last_bin.sum()
    
    return last_bin_percentage > threshold
    
def process_image_file(img_files, low_contrast_dir, threshold, queue, lock):
        image_batches = [img_files[i:i + 100] for i in range(0, len(img_files), 100)] # Batch the image files in groups of 100
        for img_batch in image_batches:
            print(f"Processing batch of {len(img_batch)} images...")
            with lock:
                wait_for_memory() # Wait for memory to be available
                images = [cv2.imread(img_file) for img_file in img_batch] # Batch read the images
            for image, img_file in zip(images, img_batch):
                try:
                    if image is not None:
                        # Convert image to grayscale
                        if len(image.shape) > 2:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        # Denoise the image
                        ahash = average_hash(Image.fromarray(image))
                        if is_low_contrast(image, threshold) or snr(image) < -18:
                            move(
                                img_file, os.path.join(
                                    low_contrast_dir, os.path.basename(img_file)))
                            print(f"Moved low contrast image: {os.path.basename(img_file)}")
                            continue
                        if is_overexposed(image):
                            move(
                                img_file, os.path.join(
                                    low_contrast_dir, os.path.basename(img_file)))
                            print(f"Moved overexposed image: {os.path.basename(img_file)}")
                            continue
                        queue.put((ahash, img_file))
                except Exception as e_img:
                    print(f"Error processing image {os.path.basename(img_file)}: {e_img}")
                    continue


def move_duplicate_img(queue, duplicate_dir: str):
    hash_set = set()
    while True:
        imhash, img_file = queue.get()
        if imhash is None:
            break  # Exit the loop if the sentinel value is received
        if imhash in hash_set:
            move(img_file, os.path.join(duplicate_dir, os.path.basename(img_file)))
        else:
            # Add the hash to the set to indicate that the image has been seen
            hash_set.add(imhash)


def move_low_contrast_images_parallel(
        image_files, low_contrast_dir, duplicate_dir, threshold=0.01, image_batch_size=1000):
    with Manager() as manager:
        lock = manager.Lock()
        queue = manager.Queue()
        duplicate_process = Process(target=move_duplicate_img, args=(queue, duplicate_dir))
        duplicate_process.start()
        image_batches = [image_files[i:i + image_batch_size] for i in range(0, len(image_files), image_batch_size)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            # Batch the image files
            try:
                futures = [
                    executor.submit(
                        process_image_file, img_batch, low_contrast_dir, threshold, queue, lock)
                    for img_batch in image_batches]
            except Exception as e:
                print(f"Error processing images: {e}")
            print(f"processing files with {len(futures)} processes")
            for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures)):
                pass  # Just wait for the futures to complete
        # Send the sentinel value to the queue to indicate that the processing is complete
        queue.put((None, None))
        duplicate_process.join() # Wait for the duplicate process to finish

def _test_over_exposed():
    img = np.random.randint(0, 255, (100, 100)).astype(np.uint8)
    img[50:60, 50:60] = 255 # Set a region to 255
    oe = is_overexposed(img, threshold=.1) # image is over exposed if the last bin has more than 10% of the pixels
    assert oe, "Failed overexposed test"
    img = np.random.randint(0, int(2**16 - 1), (100, 100)).astype(np.uint16)
    img[50:60, 50:60] = 2**16 - 1 # Set a region to the maximum intensity
    oe = is_overexposed(img, threshold=.1, max_intensity=2**16 - 1)
    assert oe, "Failed overexposed test"



if __name__ == "__main__":
    from common.helpers import DebugArgParser as ArgumentParser
    # Define the directories
    parser = ArgumentParser(
        description="Move low contrast images to a separate directory.")
    parser.add_argument("image_dir", type=str,
                        help="Directory containing the images.")
    parser.add_argument(
        "low_contrast_dir", type=str,
        help="Directory to move low contrast images.")
    parser.add_argument(
        "--threshold", type=float, default=0.01,
        help="Threshold for low contrast detection.")
    parser.add_argument(
        "--create_dir", action="store_true",
        help="Create the low contrast directory if it doesn't exist.")
    parser.add_argument(
        "--duplicate-dir", type=str, default="duplicates", dest="duplicate_dir",
        help="Directory to move duplicate images.")
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Suppress the confirmation prompt to start processing."
    )
    args = parser.parse_args()
    if not args.no_confirm:
        print("This script will move low contrast images to a separate directory. It may take a while to process the images.")
        print("Do you want to continue? (y/n)")
        response = input()
        if response.lower() != "y":
            print("Exiting...")
            exit(0)
    if args.create_dir:
        os.makedirs(args.low_contrast_dir, exist_ok=True)
        os.makedirs(args.duplicate_dir, exist_ok=True)
    print("loading files")
    image_files = glob(os.path.join(args.image_dir, "*.tif")
                       ) if os.path.isdir(args.image_dir) else glob(args.image_dir)
    print("processing images...")
    move_low_contrast_images_parallel(
        image_files, args.low_contrast_dir, args.duplicate_dir, args.threshold)
