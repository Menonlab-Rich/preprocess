import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
import multiprocessing as mp
from skimage import filters
from skimage.morphology import disk
import matplotlib.pyplot as plt


class ImageProcessor:
    def __init__(self, patch_size=(3, 3), threshold=(2**16 * 0.4, 2**16*0.7)):  # Adjusted threshold for 16-bit
        self.patch_size = patch_size
        self.threshold = threshold
        self.area = np.prod(patch_size)

    def despeckle_image(self, image):
        return filters.median(image, disk(3))
    def create_integral_image(self, image):
        # Ensure the data type of the integral image can handle the increase in range
        return cv2.integral(image, sdepth=cv2.CV_64F)[1:, 1:]

    def create_boolean_mask(self, integral_image):
        mean_image = (integral_image - np.roll(integral_image, self.patch_size[0], axis=1)
                      - np.roll(integral_image, self.patch_size[1], axis=0)
                      + np.roll(integral_image, self.patch_size, axis=(1, 0))) / self.area
        mean_image[:self.patch_size[1], :] = 0
        mean_image[:, :self.patch_size[0]] = 0
        mask = (mean_image > self.threshold[0]) & (mean_image < self.threshold[1])
        return mask.astype(np.uint8)

def process_image(args):
    img_path, processor, save_dir, review, review_dir = args
    # Load the image in its native depth (16-bit for TIFF images)
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        logging.warning(f"Failed to load image {img_path}.")
        return
    
    filtered_image = processor.despeckle_image(image)
    # Create integral image and boolean mask
    integral_image = processor.create_integral_image(filtered_image)
    mask = processor.create_boolean_mask(integral_image).astype(np.uint8) # Convert to 8-bit for storage
    img_mask = mask.copy()
    # convert the mask to an 8-bit image
    image_id = os.path.basename(img_path)
    cat_id = 1 if '625-image' in image_id else 2
    # Save the mask image; the output remains 8-bit
    mask = mask * cat_id # Set the mask to the category ID
    np.savez_compressed(os.path.join(save_dir, f"{image_id}.npz"), mask=mask, category=cat_id)
    
    if review:
        mask_img = np.zeros_like(img_mask).astype(np.uint8)  # Create an empty image
        # make the image 3 channel for visualization
        mask_img = np.dstack([mask_img] * 3) # Convert to 3-channel image
        
        # 605 is green and 625 is red
        mask_img[mask == 0] = [0, 0, 0]  # Set the mask to black
        mask_img[mask == 1] = [255, 0, 0]  # Set the mask to red for the first category
        mask_img[mask == 2] = [0, 255, 0]  # Set the mask to green for the second category
        
        # plot the images side by side for comparison
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(mask_img)
        plt.title('Mask')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(review_dir, f"{image_id}.png"))
        plt.close()
        

def mkdirs_if_not_exist(*dirs, root_dir=None, exist_ok=True):
    for subdir in dirs:
        directory = os.path.join(root_dir, subdir) if root_dir else subdir
        os.makedirs(directory, exist_ok=exist_ok)

def main():
    from glob import glob
    from common.helpers import DebugArgParser as ArgumentParser
    parser = ArgumentParser(description="Create mask images from 16-bit TIFF images.")
    parser.add_argument("img_dir", type=str, help="Directory containing the TIFF images.")
    parser.add_argument("save_dir", type=str, help="Directory to save the mask images.")
    parser.add_argument("--patch_size", type=int, nargs=2, default=(3, 3), help="Size of the patch for processing.")
    parser.add_argument("--threshold", type=int, nargs=2, default=(2560, 62565), help="Threshold for the mask.")
    parser.add_argument("--create_dir", action="store_true", help="Create the save directory if it doesn't exist.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    parser.add_argument("--num_processes", type=int, default=mp.cpu_count() // 2, help="Number of processes to use for multiprocessing.")
    parser.add_argument("--log_file", type=str, default=None, help="Path to save the log file.")
    parser.add_argument("--review", action="store_true", help="Review the mask images.")
    parser.add_argument("--review_dir", type=str, help="Directory to save the review images.")
    
    args = parser.parse_args()
    
    if args.log_file:
        logging.basicConfig(filename=args.log_file, level=args.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        # Log to stderr
        stderr_handler = logging.StreamHandler()
        logging.getLogger().addHandler(stderr_handler)
        logging.basicConfig(level=args.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.create_dir:
        mkdirs_if_not_exist(args.save_dir)
    
    processor = ImageProcessor(patch_size=args.patch_size, threshold=args.threshold)
    # Treat the input as a directory or a pattern to match 
    if os.path.isdir(args.img_dir):
        img_pattern = os.path.join(args.img_dir, '*.tif')  # Target TIFF images
    else:
        img_pattern = args.img_dir
    image_files = glob(img_pattern)

    logging.info(f"Found {len(image_files)} images to process.")

    with mp.Pool(processes=args.num_processes) as pool:
        list(tqdm(pool.imap_unordered(process_image, [(img_path, processor, args.save_dir, args.review, args.review_dir) for img_path in image_files]), total=len(image_files)))

if __name__ == '__main__':
    main()
