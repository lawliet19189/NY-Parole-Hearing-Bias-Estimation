import argparse
import pandas as pd

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from src.preprocessing import preprocess_data, engineer_features
from src.model import LogisticRegression
from src.result_viz import interpret_results, visualize_feature_importance, analyze_bias_onehot, visualize_bias
from src.utils import save_model


arg_parser = argparse.ArgumentParser(description="Train a LogisticRegression Model on the given data")
arg_parser.add_argument("--input_file", help="The input file that contains the data to train the model on")
arg_parser.add_argument("--output_dir", help="The output directory to save the trained model to")


def train_model(X, y, X_val, y_val):
    # Convert X to numeric, replacing non-numeric values with NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    X_val = X_val.apply(pd.to_numeric, errors="coerce")

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    X_val_imputed = imputer.transform(X_val)

    # Scales to Unit Variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)

    # Get feature names after imputation (in case some were dropped)
    feature_names = X.columns[~np.isnan(X_imputed).all(axis=0)]

    # A very simple model for demonstration purposes
    model = LogisticRegression(learning_rate=0.001, num_iterations=10000)
    model.fit(X_scaled, y, X_val_scaled, y_val)
    return model, imputer, scaler, feature_names


def main():
    args = arg_parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir

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

    # Train the model
    model, imputer, scaler, used_features = train_model(X_train, y_train, X_val, y_val)

    # Get feature importances
    feature_importance = interpret_results(model, used_features)

    # save the model
    save_model(model, imputer, scaler, used_features, output_dir)

    # Visualize feature importances
    plt = visualize_feature_importance(feature_importance, top_n=50, path=output_dir)

    # Analyze b& visualize bias
    race_bias_df = analyze_bias_onehot(df, "Race/Ethnicity", model, imputer, scaler, used_features)
    visualize_bias(race_bias_df, "Race/Ethnicity", path=output_dir)


if __name__ == "__main__":
    print("Starting...")
    main()
