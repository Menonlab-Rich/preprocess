def main():
    from skimage import io
    import os
    import numpy as np
    from common.helpers import DebugArgParser
    from tqdm import tqdm
    
    parser = DebugArgParser(description="Calculate statistics for a dataset.")
    parser.add_argument("input_dir", type=str, help="Input directory containing images.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for statistics.")
    
    args = parser.parse_args()
    
    images = os.listdir(args.input_dir)
    n_images = len(images)
    
    # Initialize accumulators
    sum_img = None
    sum_sq_img = None
    min_val = float('inf')
    max_val = float('-inf')
    
    for filename in tqdm(images):
        img_path = os.path.join(args.input_dir, filename)
        img = io.imread(img_path).astype(np.float64)  # Ensure using float64 for calculations
        
        if sum_img is None:
            # Initialize the sum arrays with the shape and type of the first image
            sum_img = np.zeros_like(img, dtype=np.float64)
            sum_sq_img = np.zeros_like(img, dtype=np.float64)
        
        sum_img += img
        sum_sq_img += img**2
        min_val = min(min_val, img.min())
        max_val = max(max_val, img.max())
    
    # Correct calculation of mean and std to prevent negative square roots
    mean = sum_img / n_images
    var = (sum_sq_img / n_images) - (mean**2)
    std = np.sqrt(np.maximum(var, 0))  # Avoid negative values inside sqrt
    
    # Output results
    output = f"Mean: {mean.mean()}\nStandard deviation: {std.mean()}\nMin: {min_val}\nMax: {max_val}\n"
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        stats_file = os.path.join(args.output_dir, "statistics.txt")
        with open(stats_file, "w") as f:
            f.write(output)
    else:
        print(output)

if __name__ == "__main__":
    main()
