#!/usr/bin/env python3
"""Render the human-readable freeze README from its canonical manifest."""

from argparse import ArgumentParser

from gridiron_ml.publication import render_freeze_readme


def main():
    parser = ArgumentParser()
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    print(render_freeze_readme(args.bundle))


if __name__ == "__main__":
    main()
