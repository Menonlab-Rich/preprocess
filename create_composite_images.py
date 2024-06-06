import numpy as np
from scipy.ndimage.interpolation import zoom

def gen_composite_map(size=256, zoom_factor=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Generate random points
    points = np.random.uniform(0, 2, (size, size))
    arr = zoom(points, zoom_factor)
    arr = np.array(np.round(arr), dtype=np.uint8)
    return arr

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    arr = gen_composite_map(size=3, zoom_factor=10, seed=42)
    print(arr)
    