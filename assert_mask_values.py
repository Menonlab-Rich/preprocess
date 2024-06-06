import os
import numpy as np
from glob import glob
from tqdm import tqdm

def parse_args():
    '''
    Parse the arguments passed to the script
    '''
    from common.helpers import DebugArgParser
    parser = DebugArgParser(description="Assert that the mask values are correct")
    parser.add_argument("mask_dir", type=str,
                        help="The globbing pattern to the input images")
    return parser.parse_args()

def main():
    args = parse_args()
    mask_files = glob(os.path.join(args.mask_dir, "*.npz"))
    print(f"Found {len(mask_files)} mask files")
    for mask in tqdm(mask_files):
        mask_data = np.load(mask)
        mask_values = np.unique(mask_data["mask"])
        if '625-' in mask:
            assert np.array_equal(mask_values, np.array([0, 1])), f"Mask values are incorrect for {mask}"
        elif '605-' in mask:
            assert np.array_equal(mask_values, np.array([0, 2])), f"Mask values are incorrect for {mask}"
        else:
            raise ValueError(f"Unknown mask type for {mask}")

if __name__ == "__main__":
    main()