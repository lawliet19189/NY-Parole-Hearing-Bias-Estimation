import argparse
import pandas as pd

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from src.preprocessing import preprocess_data, engineer_features
from src.model import LogisticRegression
from src.result_viz import (
    interpret_results,
    visualize_feature_importance,
    analyze_bias_onehot,
    visualize_bias,
    # visualize_race_ethnicity_importance,
    analyze_feature_importance,
    visualize_feature_importances,
)
from src.utils import load_model


arg_parser = argparse.ArgumentParser(description="Evaluate a LogisticRegression Model")
arg_parser.add_argument("--input_file", help="The input file that contains the data to train the model on")
arg_parser.add_argument("--input_dir", help="The input directory that contains the trained model")
arg_parser.add_argument("--output_dir", help="The output directory to save the evaluate results to")


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

    # Prepare features and target
    features = [col for col in df.columns if col != "Interview Decision"]
    X = df[features]
    y = df["Interview Decision"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get feature importances
    feature_importance = interpret_results(model, used_features)

    # Visualize feature importances
    # plt = visualize_feature_importance(feature_importance, top_n=50, path=output_dir)

    # Analyze b& visualize bias
    # race_bias_df = analyze_bias_onehot(df, "Race/ethnicity", model, imputer, scalar, used_features)
    # visualize_bias(race_bias_df, "Race/ethnicity", path=output_dir)

    feature_importance = analyze_feature_importance(model, used_features)
    # visualize_race_ethnicity_importance(feature_importance, df, "Race/ethnicity_")
    features_to_analyze = [
        "Race/ethnicity_",
        "Housing or Interview Facility_",
        "Parole Board Interview Type_",
        "Aggregated Minimum Sentence",
        "Aggregated Maximum Sentence",
        "Housing/Release Facility_",
        "Parole Eligibility Date",
        "Crime_",
    ]

    # Visualize importance for each feature
    for feature in features_to_analyze:
        visualize_feature_importances(feature_importance, df, feature, output_path=output_dir)
