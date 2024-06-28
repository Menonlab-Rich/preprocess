from PIL import Image
import os
from os import path
from tqdm import tqdm

def parse_args():
    from common.helpers import DebugArgParser
    parser = DebugArgParser(description='Crop a tiff file')
    parser.add_argument(
        'input_folder', type=str,
        help='Folder containing tiff files to crop')
    parser.add_argument('output_folder', type=str,
                        help='Folder to save cropped tiff files')
    parser.add_argument(
        '--crop', type=int, required=True, nargs=4, help='Crop dimensions in the format x1 y1 x2 y2')
    return parser.parse_args()

def crop_tiff(tif_image, output, crop_dims):
    with Image.open(tif_image) as img:
        img = img.crop(crop_dims)
        img.save(output)

def main():
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    crop_dims = tuple(args.crop)
    
    for tif_image in tqdm([f for f in os.listdir(input_folder) if f.endswith('.tif')]):
        tif_image = path.join(input_folder, path.basename(tif_image))
        output = path.join(output_folder, path.basename(tif_image))
        crop_tiff(tif_image, output, crop_dims)
        
if __name__ == '__main__':
    main()