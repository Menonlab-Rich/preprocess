import numpy as np
import os
import random
import tifffile as tiff
import argparse

# Pairs a mask with the image of the same name based on prefix


def load_mask_image_pairs(mask_dir, image_dir, prefix):
    '''
    Load mask-image pairs from the given directories

    Parameters
    ---
    :param mask_dir: Directory containing the masks
    :param image_dir: Directory containing the images
    :param prefix: Prefix of the image files to match with the masks

    Returns
    ---
    :return: List of tuples containing the mask and image paths
    '''
    pairs = []
    for image_file in os.listdir(image_dir):
        if image_file.startswith(prefix) and image_file.endswith('.tif'):
            mask_file = f'{os.path.basename(image_file)}.npz'
            mask_path = os.path.join(mask_dir, mask_file)
            image_path = os.path.join(image_dir, image_file)
            if os.path.exists(mask_path):
                pairs.append((mask_path, image_path))
    return pairs


def paired_tif_to_array(pairs, epoch):
    '''
    Convert a tif image in a pair to a numpy array

    Parameters
    ---
    :param pairs: List of tuples containing the mask and image paths
    :param epoch: Index of the pair to convert
    '''
    _, image_path = pairs[epoch]
    image_array = np.array(tiff.imread(image_path))
    return image_array


def get_combined_array(top_array, bottom_array, threshold):
    '''
    Combine two arrays based on a threshold. 
    The top array is used if the value is above the threshold, otherwise the bottom array is used
    Threshold will require tuning based on the data

    Parameters
    ---
    :param top_array: Array to use if the value is above the threshold
    :param bottom_array: Array to use if the value is below the threshold
    :param threshold: Threshold value to determine which array to use

    Returns
    ---
    :return: Combined array
    '''
    combined_array = np.zeros_like(
        bottom_array)  # Initialize combined array with zeros

    # Iterate over each pixel location
    for i in range(bottom_array.shape[0]):
        for j in range(bottom_array.shape[1]):
            # Check if the top array value is non-zero
            if top_array[i, j] > threshold:
                # Take value from top array
                combined_array[i, j] = top_array[i, j]
            else:
                combined_array[i, j] = bottom_array[i, j]

    return combined_array


# Layers the masks & tif images respective to the same sample number
def layer_masks_and_tifs(pairs_1, pairs_2, epoch):
    '''
    Layer masks and tif images based on the epoch number

    Parameters
    ---
    :param pairs_1: List of tuples containing the mask and image paths for the first set
    :param pairs_2: List of tuples containing the mask and image paths for the second set
    :param epoch: Index of the pair to layer

    Returns
    ---
    :return: Combined mask and array
    '''
    mask_path_1, image_path_1 = pairs_1[epoch]
    mask_path_2, image_path_2 = pairs_2[epoch]

    mask_1 = np.load(mask_path_1)['mask']
    mask_2 = np.load(mask_path_2)['mask']
    mask_2 = mask_2 / 2  # only required if both sets of the mask have the same maximum value

    image_array_1 = np.array(tiff.imread(image_path_1))
    image_array_2 = np.array(tiff.imread(image_path_2))

    if random.choice(
            [True, False]):  # Switches between what mask is on top to give random data
        top_mask = mask_1
        bottom_mask = mask_2
        top_array = image_array_1
        bottom_array = image_array_2
    else:
        top_mask = mask_2
        bottom_mask = mask_1
        top_array = image_array_2
        bottom_array = image_array_1

    combined_mask = np.where(top_mask != 0, top_mask, bottom_mask)
    # This threshold value doesnt pick up light under about 23% intensity ((2**16 - 1) * .22889)
    combined_array = get_combined_array(top_array, bottom_array, 15000)
    # This creates a loss around the edges, but allows layers to appear more accurate
    # change according to specific testing
    # TODO: Make this an argument to the script
    return combined_mask, combined_array


def save_files(combined_array, combined_mask, epoch, folder_output):
    '''
    Save the combined array as a tif and the combined mask as a npz
    Creates the output directories if they don't exist

    Parameters
    ---
    :param combined_array: Array to save as a tif
    :param combined_mask: Mask to save as a npz
    :param epoch: Epoch number to use in the file name
    :param folder_output: Folder to save the output files
    '''
    # Create output directories if they don't exist
    combined_tifs_dir = os.path.join(folder_output, 'combined_tifs')
    combined_masks_dir = os.path.join(folder_output, 'combined_masks')
    os.makedirs(combined_tifs_dir, exist_ok=True)
    os.makedirs(combined_masks_dir, exist_ok=True)

    # Save combined_array as tif
    tif_output_path = os.path.join(combined_tifs_dir, f'{epoch}_tif.tif')
    tiff.imwrite(tif_output_path, combined_array)
    # Save combined_mask as npz
    npz_output_path = os.path.join(combined_masks_dir, f'{epoch}_mask.npz')
    np.savez_compressed(npz_output_path, mask=combined_mask)


def parse_args():
    try:
        from common.helpers import DebugArgParser as ArgumentParser
    except ImportError:
        ArgumentParser = argparse.ArgumentParser

    parser = ArgumentParser(
        description='Combine masks and tif images based on the same sample number')
    parser.add_argument('--mask-dir', type=str,
                        help="Directory containing the masks")
    parser.add_argument('--image-dir', type=str,
                        help="Directory containing the images")
    parser.add_argument(
        '--output-dir', type=str,
        help="Folder that will contain the output folders 'combined_masks' and 'combined_tifs'")
    return parser.parse_args()


def main():
    args = parse_args()

    pairs_1 = load_mask_image_pairs(
        mask_dir=args.mask_dir, image_dir=args.image_dir, prefix='625')
    pairs_2 = load_mask_image_pairs(
        mask_dir=args.mask_dir, image_dir=args.image_dir, prefix='605')
    epoch_1 = len(pairs_1)
    epoch_2 = len(pairs_2)

    num_epoch = min(epoch_1, epoch_2)

    for epoch in range(num_epoch):  # Repeats equal to the lesser amount of samples
        combined_mask, combined_array = layer_masks_and_tifs(
            pairs_1, pairs_2, epoch)
        save_files(combined_array, combined_mask, epoch, args.output_dir)
        if (epoch + 1) % 5 == 0:
            print(f"Finished Epoch: {epoch + 1}")


if __name__ == "__main__":
    main()
