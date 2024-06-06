import numpy as np
from PIL import Image
from glob import glob

from PIL import Image
import numpy as np
import glob

def make_df(img_pattern, output_file="dark_frame.tif", crop_dims=None):
    '''
    Make a dark frame from a set of images matching the given pattern, specifically handling 16-bit images.
    '''
    input_files = sorted(glob.glob(img_pattern))
    dark_frame = None

    for i, input_file in enumerate(input_files):
        img = Image.open(input_file)
        if crop_dims is not None:
            img = crop(img, crop_dims) # Crop the image to the given dimensions
        if i == 0:
            # Initialize dark_frame as a float array to accumulate pixel values accurately
            dark_frame = np.zeros(np.array(img).shape, dtype=np.float32)
        
        dark_frame += np.array(img, dtype=np.float32)
    
    if dark_frame is not None:
        # Average the images
        dark_frame /= len(input_files)
        
        # Convert the averaged frame back to 16-bit unsigned integer
        dark_frame = np.clip(dark_frame, 0, 65535).astype(np.uint16)
        
        # Create an image from the array. Assume grayscale ('I;16') if single channel, adjust as necessary for RGB
        mode = 'I;16' if dark_frame.ndim == 2 else 'RGB'
        dark_frame_img = Image.fromarray(dark_frame, mode)
        
        # Save the dark frame
        dark_frame_img.save(output_file)
        
def crop(img, dimensions):
    '''
    Crop an image to the given dimensions.
    '''
    w, h = img.size
    expected_w, expected_h = dimensions
    left = (w - expected_w) // 2
    top = (h - expected_h) // 2
    right = (w + expected_w) // 2
    bottom = (h + expected_h) // 2
    return img.crop((left, top, right, bottom))

def main():
    from common.helpers import DebugArgParser
    parser = DebugArgParser()
    parser.add_argument("img_pattern", type=str, help="Glob pattern to match the input images.")
    parser.add_argument("output_file", type=str, default="dark_frame.tif", help="Output file for the dark frame.")
    parser.add_argument("--crop", type=int, nargs=2, default=None, help="Crop the images to the given dimensions before averaging.")
    args = parser.parse_args()
    make_df(args.img_pattern, args.output_file, args.crop)
    
if __name__ == "__main__":
    main()