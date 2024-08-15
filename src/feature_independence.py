import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.utils import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


from src.preprocessing import preprocess_data, engineer_features


arg_parser = argparse.ArgumentParser(description="Evaluate a LogisticRegression Model")
arg_parser.add_argument("--input_file", help="The input file that contains the data to train the model on")
arg_parser.add_argument("--input_dir", help="The input directory that contains the trained model")
arg_parser.add_argument("--output_dir", help="The output directory to save the evaluate results to")


def analyze_feature_independence(df, target_column, path):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle NaN values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Correlation analysis
    corr_matrix = X_imputed.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(path + "/correlation_heatmap.png")
    plt.close()

    # Mutual Information
    mi_scores = mutual_info_classif(X_imputed, y)
    mi_df = pd.DataFrame({"Feature": X_imputed.columns, "Mutual Information": mi_scores})
    mi_df = mi_df.sort_values("Mutual Information", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Mutual Information", y="Feature", data=mi_df)
    plt.title("Mutual Information with Target Variable")
    plt.tight_layout()
    plt.savefig(path + "/mutual_information.png")
    plt.close()

    # VIF (for numerical features)
    numerical_features = X_imputed.select_dtypes(include=[np.number]).columns
    X_scaled = StandardScaler().fit_transform(X_imputed[numerical_features])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numerical_features
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    vif_data = vif_data.sort_values("VIF", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="VIF", y="Feature", data=vif_data)
    plt.title("Variance Inflation Factor for Numerical Features")
    plt.tight_layout()
    plt.savefig(path + "/vif_analysis.png")
    plt.close()

    return corr_matrix, mi_df, vif_data


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

    corr_matrix, mi_df, vif_data = analyze_feature_independence(df, "Interview Decision", path=output_dir)

    # Print top correlated features
    print("Top Correlated Features:")
    print(corr_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(10))

    print("\nTop Features by Mutual Information:")
    print(mi_df.head(10))

    print("\nFeatures with High VIF (>5):")
    print(vif_data[vif_data["VIF"] > 5])
