from glob import glob
from PIL import Image
import os



def process_args():
    from common.helpers import DebugArgParser as ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('input_dir', help='Directory containing the 16-bit TIFF images.')

    return parser.parse_args()

def main():
    args = process_args()
    input_dir = args.input_dir
    tif_files = glob(os.path.join(input_dir, '*.tif'))
    broken_tifs = []
    for tif_file in tif_files:
        try:
            Image.open(tif_file)
        except Exception as e:
            broken_tifs.append(tif_file)
    for broken_tif in broken_tifs:
        print(f"Removing broken TIFF: {broken_tif}")
        os.remove(broken_tif)

if __name__ == '__main__':
    main()