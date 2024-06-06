def process_args():
    from common import helpers
    parser = helpers.DebugArgParser()
    parser.add_argument("--sample_size", type=int, default=5)
    parser.add_argument("--source_dir", type=str, default="data/raw")
    parser.add_argument("--target_dir", type=str, default="data/sample")
    parser.add_argument("--prefix", type=str, default="sample")
    args = parser.parse_args()
    return args


def main():
    args = process_args()
    copy_sample(args.sample_size, args.source_dir, args.target_dir, args.prefix)


def copy_sample(
        sample_size: int, source_dir: str, target_dir: str, prefix: str):
    import os
    import shutil
    import random
    files = os.listdir(source_dir)
    files_to_copy = [f for f in files if f.startswith(prefix)]
    files_to_copy = random.sample(files_to_copy, sample_size)
    for f in files_to_copy:
        shutil.copy(os.path.join(source_dir, f), target_dir)
    print(
        f"Successfully copied {sample_size} files from {source_dir} to {target_dir}")


if __name__ == '__main__':
    main()