import argparse
import sys
from .orchestrator import process_file, process_folder, to_json


def main():
    ap = argparse.ArgumentParser(description="Local IDP Prototype")
    ap.add_argument("path", help="Path to a file or folder of .txt docs")
    args = ap.parse_args()
    target = args.path
    if not target:
        print("Provide a path to a file or folder", file=sys.stderr)
        sys.exit(1)
    import os

    if os.path.isdir(target):
        results = process_folder(target)
        for r in results:
            print(to_json(r))
            print()
    else:
        r = process_file(target)
        print(to_json(r))


if __name__ == "__main__":
    main()

