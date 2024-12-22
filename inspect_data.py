"""Print a summary of a summary data file."""

import os
import sys
import pandas as pd


def main(args):
    if args.fname is not None and os.path.isfile(args.fname):
        fname_base = os.path.basename(args.fname)
        name, ext = os.path.splitext(fname_base)
        if ext == ".pkl":
            # We are dealing with a pickled dataframe
            df = pd.read_pickle(args.fname)
            print(df)
        else:
            # We don't know what to do
            print(f"Invalid file '{args.fname}'", file=sys.stderr)
    else:
        print(f"File '{args.fname}' does not exist", file=sys.stderr)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, help="file to inspect")

    args = parser.parse_args()
    main(args)
