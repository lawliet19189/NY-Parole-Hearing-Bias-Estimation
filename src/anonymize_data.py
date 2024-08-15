import randomname
import pandas as pd
import argparse
import numpy as np

arg_parser = argparse.ArgumentParser(description="Anonymize data")
arg_parser.add_argument("--input_file", help="The input file to anonymize")
arg_parser.add_argument("--output_file", help="The output file to write the anonymized data to")

existing_randoms = set()


def generate_random_name():
    while True:
        random_name = randomname.generate(
            "adj/music_theory",
            ("nouns/*"),
            ("n/cats", "n/food", "n/fruit", "n/apex_predators", "n/ghosts", "n/fish", "n/plants", "n/cheese"),
        )

        if random_name not in existing_randoms:
            existing_randoms.add(random_name)
            return random_name


def generate_random_number():
    while True:
        rng = np.random.default_rng()
        rints = rng.integers(low=1, high=100000, size=1)[0]
        if rints not in existing_randoms:
            existing_randoms.add(rints)
            return rints


def main():
    args = arg_parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    df = pd.read_csv(input_file)

    for col in df.columns:
        if "name" in col.lower():
            df[col] = df[col].apply(lambda x: generate_random_name())

        # update 'DIN' column to random number
        if "DIN" in col:
            df[col] = df[col].apply(lambda x: generate_random_number())

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
