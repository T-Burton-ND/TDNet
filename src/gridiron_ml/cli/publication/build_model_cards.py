#!/usr/bin/env python3
"""Render model cards from the final frozen-model inventory."""

from argparse import ArgumentParser
import pandas as pd

from gridiron_ml.publication import build_model_cards


def main():
    parser = ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cards = build_model_cards(pd.read_csv(args.inventory), args.output_dir)
    print(f"cards={len(cards)}")


if __name__ == "__main__":
    main()
