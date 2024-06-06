import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

def zip_inputs_and_targets(input_pattern, target_pattern):
    input_files = sorted(glob.glob(input_pattern))
    target_files = sorted(glob.glob(target_pattern))
    return zip(input_files, target_files)

def examine_shapes(input_pattern, target_pattern):
    for input_file, target_file in tqdm(zip_inputs_and_targets(input_pattern, target_pattern)):
        target = TARGET_READER(target_file, 0)
        print(len(np.unique(target)))
        input_image = INPUT_READER(input_file, 1)
        # check the spatial dimensions of the input and target images
        if input_image.squeeze().shape != target.shape:
            print(f"Input shape: {input_image.squeeze().shape}, Target shape: {target.shape}")
            print(f"Input file: {input_file}, Target file: {target_file}")
            break
    
    print("All input and target images have the same spatial dimensions")
        
def TARGET_READER(path: str, _: int):
    # Load the mask from a .npy file
    x = np.load(path)['mask']

    # Map class labels: -1 -> 1 (class of interest 1), 0 -> 0 (background), 1 -> 2 (class of interest 2)
    target_mapped = np.where(x == -1, 1, np.where(x == 0, 0, 2))

    return target_mapped

def INPUT_READER(x, channels):
    img = Image.open(x)
    if img.mode in ["I", "I;16", "I;16B", "I;16L"]:  # For 16-bit grayscale images
        if channels == 3:  # If expecting RGB output, convert accordingly
            img = img.convert("RGB")
        else:  # Keep as is for grayscale
            img = img.convert("I")
    elif img.mode not in ["RGB", "L"]:  # If not standard 8-bit modes, convert
        img = img.convert("RGB" if channels == 3 else "L")
    return np.array(img)

def main():
    from common.helpers import DebugArgParser
    parser = DebugArgParser()
    parser.add_argument("--input_pattern", type=str, required=True)
    parser.add_argument("--target_pattern", type=str, required=True)
    args = parser.parse_args()
    examine_shapes(args.input_pattern, args.target_pattern)
    
if __name__ == "__main__":
    main()