#!/usr/bin/env python3
from argparse import ArgumentParser
from gridiron_ml.publication.selection import write_confirmatory_roster

def main():
    parser = ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    roster = write_confirmatory_roster(args.candidates, args.output)
    print(roster[["roster_rank", "concrete_model_type", "selection_brier_score", "publication_role"]].to_string(index=False))

if __name__ == "__main__":
    main()
