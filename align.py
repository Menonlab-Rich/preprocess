from common.helpers import DebugArgParser
from os import path, listdir
from tqdm import tqdm
import skimage as sk
from skimage import io, color
import numpy as np

def norm(img):
    img = img.astype('float64')
    return (img - img.min()) / (img.max() - img.min()) # Normalize the image to [0, 1]

def convert_to_rgb(img):
    if len(img.shape) == 2 or img.shape[2] == 1: # If the image is grayscale
        img = color.gray2rgb(img) # Convert grayscale to RGB
    return img

def convert_mask_to_rgb(mask, rgb_mapping):
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for value, rgb in rgb_mapping.items():
        rgb_mask[mask == value] = rgb
    return rgb_mask

def save(img, path, ext='.npz'):
    if ext == '.npz':
        np.savez(path, img)
    elif ext == '.npy':
        np.save(path, img)
    else:
        io.imsave(path, img, check_contrast=False)

def align(image, mask, output, mask_ext='.npz', mask_key='mask', image_ext=None, output_ext='-combined.tiff', rgb_mapping=None):
    img = norm(io.imread(image)) # Read the image and normalize it
    img = convert_to_rgb(img) # Convert the image to RGB
    if mask_ext == '.npz':
        mask = np.load(mask)[mask_key] # Read the mask
    elif mask_ext == '.npy':
        mask = np.load(mask)
    else:
        mask = io.imread(mask)
    
    mask = mask.astype('float64') # Convert the mask to float64

    mask = convert_mask_to_rgb(mask, rgb_mapping) # Convert the mask to RGB using the mapping
    
    # Pad the smaller image to the size of the larger image
    if img.shape[0] > mask.shape[0]:
        mask = np.pad(mask, ((0, img.shape[0] - mask.shape[0]), (0, 0), (0, 0)), mode='constant')
    elif img.shape[0] < mask.shape[0]:
        img = np.pad(img, ((0, mask.shape[0] - img.shape[0]), (0, 0), (0, 0)), mode='constant')
    
    if img.shape[1] > mask.shape[1]:
        mask = np.pad(mask, ((0, 0), (0, img.shape[1] - mask.shape[1]), (0, 0)), mode='constant')
    elif img.shape[1] < mask.shape[1]:
        img = np.pad(img, ((0, 0), (0, mask.shape[1] - img.shape[1]), (0, 0)), mode='constant')
    
    # Combine the image and mask
    img = np.squeeze(img * 255).astype(np.uint8) # Remove any singleton dimensions
    mask = np.squeeze(mask)# Remove any singleton dimensions
    # Concatenate the image and mask along the vertical edge
    combined = np.concatenate((img, mask), axis=1)
    save(combined, output, output_ext) # Save the combined image and mask
    

def main():
    args = DebugArgParser(description='Aligns the Mask and Image')
    args.add_argument('image', help='Path to the image')
    args.add_argument('mask', help='Path to the mask')
    args.add_argument('output', help='Path to the output')
    args.add_argument('--mask-ext', help='Extension of the mask files', default='.npz', dest='mask_ext')
    args.add_argument('--image-ext', help='Extension of the image files', default=None, dest='image_ext')
    args.add_argument('--output-ext', help='Extension of the output files', default='-combined.tif', dest='output_ext')
    args.add_argument('--mask-key', help='Key of the mask in the npz file', default='mask', dest='mask_key')
    
    args = args.parse_args() # Parse the arguments
    valid_exts = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    
    # Create a consistent RGB mapping for mask values
    rgb_mapping = {
        0: [0, 0, 0], # Background
        1: [255, 0, 0], # 625nm
        2: [0, 255, 0], # 605nm
    }
    
    for f in tqdm(listdir(args.image)):
        ext = f.split('.')[-1]
        if args.image_ext is not None and not ext.lower() == args.image_ext.lower():
            continue
        
        if args.image_ext is None and not ext.lower() in valid_exts:
            print(f'Invalid extension: {ext}')
            continue
        
        img_path = path.join(args.image, path.basename(f)) # Get the image path
        mask_path = path.join(args.mask, path.basename(f) + args.mask_ext) # Get the mask path
        out_path = path.join(args.output, '.'.join(path.basename(f).split('.')[:-1]) + args.output_ext) # Get the output path
        
        align(img_path, mask_path, out_path, args.mask_ext, mask_key=args.mask_key, image_ext=args.image_ext, output_ext=args.output_ext, rgb_mapping=rgb_mapping)
    
if __name__ == "__main__":
    main()
