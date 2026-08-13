#!/usr/bin/env python3
"""Regenerate a freeze bundle's portable SHA256SUMS inventory."""

from argparse import ArgumentParser

from gridiron_ml.publication import write_sha256sums


def main():
    parser = ArgumentParser()
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    print(write_sha256sums(args.bundle))


if __name__ == "__main__":
    main()
