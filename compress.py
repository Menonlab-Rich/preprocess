from PIL import Image
import os
from os import path
from tqdm import tqdm


def process_args():
    from common.helpers import DebugArgParser
    parser = DebugArgParser(description='Compress a tiff file')
    parser.add_argument(
        'input_folder', type=str,
        help='Folder containing tiff files to compress')
    parser.add_argument('output_folder', type=str,
                        help='Folder to save compressed tiff files')
    parser.add_argument(
        '--compression', type=str, default='tiff_adobe_deflate',
        help='Compression type (default: jpeg)')
    return parser.parse_args()


def assert_compression_type(compression):
    possible_compressions = [
        "raw",
        "tiff_ccitt",
        "group3",
        "group4",
        "tiff_lzw",
        "tiff_jpeg",  # obsolete
        "jpeg",
        "tiff_adobe_deflate",
        "tiff_raw_16",  # 16-bit padding
        "packbits",
        "tiff_thunderscan",
        "tiff_deflate",
        "tiff_sgilog",
        "tiff_sgilog24",
        "lzma",
        "zstd",
        "webp",
    ]

    assert compression in possible_compressions, \
        f"Invalid compression type: {compression}"


def compress_tiff(tif_image, output, compression):
    assert_compression_type(compression)
    with Image.open(tif_image) as img:
        img.save(output, compression=compression)

def main():
    args = process_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    compression = args.compression

    for tif_image in tqdm([f for f in os.listdir(input_folder) if f.endswith('.tif')]):
        tif_image = path.join(input_folder, path.basename(tif_image))
        output = path.join(output_folder, path.basename(tif_image))
        compress_tiff(tif_image, output, compression)
        
if __name__ == '__main__':
    main()
        
