import numpy as np
import tifffile as tiff
import os
import random
import cv2


def load_mask_image_pairs(mask_dir, image_dir):
    pairs = []
    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.npz'):
            image_file = '.'.join(os.path.basename(mask_file).split('.')[:-1])
            mask_path = os.path.join(mask_dir, mask_file)
            image_path = os.path.join(image_dir, image_file)
            if os.path.exists(image_path):
                pairs.append((mask_path, image_path))
    return pairs


def sample_patch(mask, image, patch_size=(32, 32)):
    max_y, max_x = mask.shape[0] - patch_size[0], mask.shape[1] - patch_size[1]
    x, y = random.randint(0, max_x), random.randint(0, max_y)
    mask_patch = mask[y:y+patch_size[0], x:x+patch_size[1]]
    image_patch = image[y:y+patch_size[0], x:x+patch_size[1]]
    return mask_patch, image_patch


def create_composite(
        pairs, patch_size=(32, 32),
        composite_size=(512, 512),
        num_patches=100, photometric='rgb'):
    composite_image = np.zeros(
        list(composite_size) + [3,],
        dtype=np.uint16) if photometric == 'rgb' else np.zeros(
        composite_size, dtype=np.uint16)
    composite_mask = np.zeros(composite_size, dtype=np.int8)
    for _ in range(num_patches):
        mask_path, image_path = random.choice(pairs)
        with np.load(mask_path) as data:
            mask = data['mask']
            category = data['category'].item()
        image = tiff.imread(image_path)
        _, image_patch = sample_patch(mask, image, patch_size)
        # Random position on the composite image
        x, y = random.randint(
            0, composite_size[1]-patch_size[1]), random.randint(0, composite_size[0]-patch_size[0])
        composite_image[y:y+patch_size[0], x:x+patch_size[1]] = image_patch
        composite_mask[y:y+patch_size[0], x:x+patch_size[1]] = category
    return composite_mask, composite_image


def verify_mask(mask, mask_name):
    mask_img = np.zeros_like(mask).astype(np.uint8)  # Create an empty image
    mask_img[mask == -1] = 255  # Set the mask to white
    mask_img[mask == 1] = 128  # Set the mask to gray
    tiff.imwrite(f'{mask_name}.tif', mask_img,
                 photometric='minisblack')  # Save the mask image


def parse_args():
    from common.helpers import DebugArgParser
    parser = DebugArgParser(
        description="Generate composite image and mask from mask-image pairs.")

    # Mutually exclusive groups for patch sizing and number of patches
    patch_size_group = parser.add_mutually_exclusive_group()
    patch_num_group = parser.add_mutually_exclusive_group()

    # Mandatory arguments for directories
    parser.add_argument("mask_dir", type=str,
                        help="Directory containing the mask files.")
    parser.add_argument("image_dir", type=str,
                        help="Directory containing the image files.")
    parser.add_argument(
        '--output_dir', type=str, default='output',
        help="Output directory for the composite images and masks.")

    # Patch size arguments
    patch_size_group.add_argument(
        "--patch_size", type=int, nargs=2, default=[32, 32],
        help="Size of the patches to sample.")
    patch_size_group.add_argument(
        '--max_patch_size', type=int, nargs=2,
        help="Maximum size of patches to sample, requires --min_patch_size.")
    parser.add_argument(
        '--min_patch_size', type=int, nargs=2,
        help="Minimum size of patches to sample, requires --max_patch_size.")

    # Composite image size
    parser.add_argument(
        "--composite_size", type=int, nargs=2, default=[512, 512],
        help="Size of the composite image.")

    # Patch number arguments
    patch_num_group.add_argument(
        "--num_patches", type=int, default=100,
        help="Exact number of patches to sample.")
    patch_num_group.add_argument(
        '--max_patches', type=int,
        help="Maximum number of patches to sample, requires --min_patches.")
    parser.add_argument(
        '--min_patches', type=int,
        help="Minimum number of patches to sample, requires --max_patches.")

    # Additional arguments
    parser.add_argument(
        "--photometric", type=str, default='rgb',
        help="Photometric interpretation of the composite image.")
    parser.add_argument(
        '--verify_mask', action='store_true',
        help="Verify the generated mask by saving it as an image.",
        default=False)
    parser.add_argument('--imgs', type=int, default=1000,
                        help="Number of images to generate.")

    args = parser.parse_args()

    # Custom logic for enforcing conditional requirements
    if args.max_patches and not args.min_patches:
        parser.error("min_patches is required when max_patches is set.")
    if args.min_patches and not args.max_patches:
        parser.error("max_patches is required when min_patches is set.")
    if args.max_patch_size and not args.min_patch_size:
        parser.error("min_patch_size is required when max_patch_size is set.")
    if args.min_patch_size and not args.max_patch_size:
        parser.error("max_patch_size is required when min_patch_size is set.")

    return args


def main():
    from tqdm import tqdm
    import os
    args = parse_args()
    # Handle randomization of patch size and number of patches if required
    for i in tqdm(range(args.imgs)):
        if args.max_patches:
            args.num_patches = random.randint(
                args.min_patches, args.max_patches)
        if args.max_patch_size:
            args.patch_size = (
                random.randint(args.min_patch_size[0], args.max_patch_size[0]),
                random.randint(args.min_patch_size[1], args.max_patch_size[1]))
        
        # Create the output directory if it doesn't exist 
        os.makedirs(os.path.join(args.output_dir, 'tifs'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'masks'), exist_ok=True)
        
        # Load mask-image pairs and create composite image and mask
        pairs = load_mask_image_pairs(args.mask_dir, args.image_dir)
        composite_mask, composite_image = create_composite(
            pairs, patch_size=args.patch_size, composite_size=args.composite_size,
            num_patches=args.num_patches, photometric=args.photometric)
        
        blur_kernel = np.ones((3, 3), np.float32) / 9 # Create a 3x3 blur kernel
        # blur the composite image
        #composite_image = cv2.filter2D(composite_image, -1, blur_kernel)

        # Save the composite image and mask
        tiff.imwrite(
            os.path.join(args.output_dir, 'tifs', f'composite_{i}.tif'),
            composite_image, photometric=args.photometric)
        np.savez(os.path.join(args.output_dir, 'masks',
                 f'composite_{i}.npz'), mask=composite_mask)

        # Save the verification mask image if required
        if args.verify_mask:
            verify_mask(os.path.join(args.output_dir, 'masks', f'composite_mask_{i}.tif'))


if __name__ == "__main__":
    main()