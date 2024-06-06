from common.helpers import DebugArgParser
import numpy as np

def process_args():
    parser = DebugArgParser(description='Calculate the ratio of files matching a given filter in a directory')
    parser.add_argument('dir', help='Directory to process')
    parser.add_argument('-f', '--filters', help="Glob patterns used to split the files", nargs='+', default=["*"])
    return parser.parse_args()

def main():
    args = process_args()
    from os import listdir, path
    from glob import glob
    
    total_files = len(listdir(args.dir))
    types = []
    
    for f in args.filters:
        # Get the files that match the filter
        files = glob(path.join(args.dir, f))
        types.append(len(files))
    
    total_matched = sum(types)
    if total_matched == 0:
        print("No files matched the filters.")
        return

    # Calculate the ratios
    ratios = [t / total_matched for t in types]
    
    # Calculate weights
    weights = 1 / np.array(ratios)
    weights /= weights.sum()  # Normalize the weights
    
    print(f"Total files in directory: {total_files}")
    print(f"Total files matching filters: {total_matched}")
    
    for i, f in enumerate(args.filters):
        print(f"Files matching filter {f}: {types[i]} ({ratios[i]:.2%})")
        print(f"Suggested weighting: {weights[i]:.2f}")
        
if __name__ == '__main__':
    main()
