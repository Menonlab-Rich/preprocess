def parse_args():
    from common.helpers import DebugArgParser as ArgumentParser
    parser = ArgumentParser(description='Pad tif files')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--output-size', type=int, required=True, help='Output size (width x height)', nargs=2)
    return parser.parse_args()

def pad(image, output_size):
    from PIL import Image
    from common.helpers import pad_image
    import numpy as np
    image = Image.open(image)
    image = Image.fromarray(pad_image(np.array(image), output_size))
    return image

def main():
    import os
    from tqdm import tqdm
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_size = args.output_size[::-1] # PIL uses (width, height) instead of (height, width)
    files = [os.path.join(input_dir, os.path.basename(img)) for img in os.listdir(input_dir)]
    for img in tqdm(files):
        output_path = os.path.join(output_dir, os.path.basename(img))
        padded = pad(img, output_size)
        padded.save(output_path)
        
if __name__ == '__main__':
    main()