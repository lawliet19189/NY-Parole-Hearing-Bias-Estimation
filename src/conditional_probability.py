import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils import load_model


from src.preprocessing import preprocess_data, engineer_features


arg_parser = argparse.ArgumentParser(description="Evaluate a LogisticRegression Model")
arg_parser.add_argument("--input_file", help="The input file that contains the data to train the model on")
arg_parser.add_argument("--input_dir", help="The input directory that contains the trained model")
arg_parser.add_argument("--output_dir", help="The output directory to save the evaluate results to")


def calculate_marginal_probabilities(
    df, model, imputer, scalar, used_features, categorical_features, output_path, n_samples=10000
):
    os.makedirs(output_path, exist_ok=True)

    sample_df = pd.DataFrame(df, columns=used_features)

    # Preprocess the sample data
    sample_df_imputed = pd.DataFrame(imputer.transform(sample_df), columns=used_features)
    sample_df_scaled = pd.DataFrame(scalar.transform(sample_df_imputed), columns=used_features)

    results = {}

    for feature_prefix in categorical_features:
        feature_columns = [col for col in used_features if col.startswith(feature_prefix)]

        if not feature_columns:
            print(f"No columns found for feature prefix: {feature_prefix}")
            continue

        feature_probs = []

        for feature in feature_columns:
            # Set the current feature to 1, keeping other features as they are
            temp_df = sample_df_scaled.copy()
            temp_df[feature] = 1

            # For mutually exclusive features, set other features in the same category to 0
            for other_feature in feature_columns:
                if other_feature != feature:
                    temp_df[other_feature] = 0

            # Predict probabilities
            probs = model.predict_proba(temp_df)

            # Calculate the average probability
            avg_prob = np.mean(probs)

            feature_probs.append((feature.replace(feature_prefix, ""), avg_prob))

        # Sort probabilities
        feature_probs.sort(key=lambda x: x[1], reverse=True)

        results[feature_prefix] = feature_probs

        # Plotting
        plt.figure(figsize=(12, 8))
        categories, probs = zip(*feature_probs)
        sns.barplot(x=list(probs), y=list(categories))

        plt.title(f"Marginal Probability of Parole Given {feature_prefix}")
        plt.xlabel("Probability of Parole")
        plt.ylabel(feature_prefix)
        plt.xlim(0, 1)
        plt.tight_layout()

        # Save the plot
        plt.savefig(f'{output_path}/marginal_prob_{feature_prefix.replace("/", "_")}.png')
        plt.close()

        # Print the probabilities
        print(f"\nMarginal Probabilities for {feature_prefix}:")
        for category, prob in feature_probs:
            print(f"{category}: {prob:.4f}")

    return results


if __name__ == "__main__":
    args = arg_parser.parse_args()
    input_file = args.input_file
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Load the trained model
    model, imputer, scalar, used_features = load_model(input_dir)

    # load the data
    df = pd.read_csv(input_file)

    # drop rows with unneceeary data
    df.drop(columns=["Release Type", "Release Date"], inplace=True)

    # preprocess & engineer the data
    df = preprocess_data(df)
    df = engineer_features(df)

    categorical_features = [
        "Race/ethnicity_",
        "Housing or Interview Facility_",
        "Parole Board Interview Type_",
        "Housing/Release Facility_",
        "Crime",
    ]

    df.drop(columns=["Interview Decision"])

    marginal_probs = calculate_marginal_probabilities(
        df, model, imputer, scalar, used_features, categorical_features, output_dir
    )
