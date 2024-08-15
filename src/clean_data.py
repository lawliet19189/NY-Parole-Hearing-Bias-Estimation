import pandas as pd
import argparse
import numpy as np

arg_parser = argparse.ArgumentParser(description="Clean data")
arg_parser.add_argument("--input_file", help="The input file to anonymize")
arg_parser.add_argument("--output_file", help="The output file to write the anonymized data to")


def main():
    args = arg_parser.parse_args()
    input_file = args.input_file
    if not args.output_file:
        output_file = input_file
    else:
        output_file = args.output_file

    df = pd.read_csv(input_file)
    new_df = pd.DataFrame(columns=df.columns.to_list())

    for row_idx, row in df.iterrows():
        if row["Interview Decision"] in (
            "*",
            "**********",
            "OTHER",
            "RCND&HOLD",
            "REINSTATE",
            "RCND&RELSE",
            "RCND&HOLD",
        ):
            continue

        # if Interview Decision means parole, just change it to 'PAROLE'
        if row["Interview Decision"] in ("GRANTED", "OPEN DT", "PAROLED", "OR EARLIER", "OPEN DATE", "PAROLED"):
            row["Interview Decision"] = "PAROLE"
        else:
            row["Interview Decision"] = "DENIED"

        # add the row to the new dataframe
        new_df.loc[new_df.shape[0]] = row.to_list()

    new_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
