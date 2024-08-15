import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np


from sklearn.metrics import confusion_matrix


def interpret_results(model, feature_names):
    """Interpret the results of the model."""
    feature_importance = pd.DataFrame({"feature": feature_names, "weight": model.weights})
    feature_importance["abs_weight"] = abs(feature_importance["weight"])
    feature_importance = feature_importance.sort_values("abs_weight", ascending=False)
    return feature_importance


def visualize_feature_importance(feature_importance, top_n=20, path="."):
    plt.figure(figsize=(12, 10))

    # Select top N features by absolute weight
    top_features = feature_importance.head(top_n)

    # Create color palette
    colors = sns.color_palette("RdYlBu", n_colors=len(top_features))
    colors = [colors[i] if w >= 0 else colors[-i - 1] for i, w in enumerate(top_features["weight"])]

    # Create the plot
    ax = sns.barplot(x="weight", y="feature", data=top_features, palette=colors)

    # Customize the plot
    plt.title(f"Top {top_n} Feature Importances in Logistic Regression Model", fontsize=16)
    plt.xlabel("Weight", fontsize=12)
    plt.ylabel("Feature", fontsize=12)

    # Add weight values on the bars
    for i, v in enumerate(top_features["weight"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=10)

    # Add a vertical line at x=0
    plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)

    # Adjust layout and display
    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + "/feature_importance.png")
    plt.show()

    return plt


def analyze_bias_onehot(df, feature_prefix, model, imputer, scaler, used_features):
    # Identify columns related to the feature
    feature_columns = [col for col in df.columns if col.startswith(feature_prefix)]

    # Prepare features and target
    X = df[used_features]
    y = df["Interview Decision"].values

    # Impute and scale features
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    # Get predictions
    y_pred = model.predict(X_scaled)

    results = []

    for feature in feature_columns:
        group = feature.split("_")[-1]  # Extract the category name
        mask = df[feature] == 1
        y_true_group = y[mask]
        y_pred_group = y_pred[mask]

        if len(y_true_group) > 0:
            try:
                cm = confusion_matrix(y_true_group, y_pred_group)
                if cm.size == 1:  # Only one class present
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                elif cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                else:
                    raise ValueError("Unexpected confusion matrix shape")

                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                results.append(
                    {
                        "Group": group,
                        "Count": mask.sum(),
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1-score": f1,
                    }
                )
            except Exception as e:
                print(f"Error processing group {group}: {str(e)}")
                results.append(
                    {
                        "Group": group,
                        "Count": mask.sum(),
                        "Accuracy": np.nan,
                        "Precision": np.nan,
                        "Recall": np.nan,
                        "F1-score": np.nan,
                    }
                )

    return pd.DataFrame(results)


def visualize_bias(bias_df, feature_name, path="."):
    plt.figure(figsize=(12, 6))

    # Melt the DataFrame for easier plotting
    melted_df = bias_df.melt(
        id_vars=["Group", "Count"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
        var_name="Metric",
        value_name="Value",
    )

    # Create the grouped bar plot
    sns.barplot(x="Group", y="Value", hue="Metric", data=melted_df)

    plt.title(f"Model Performance Across {feature_name} Groups")
    plt.xlabel(feature_name)
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + f"/bias_{feature_name.replace('/', '-')}_metrics.png")
    plt.show()

    # Create a pie chart for group distribution
    plt.figure(figsize=(8, 8))
    plt.pie(bias_df["Count"], labels=bias_df["Group"], autopct="%1.1f%%")
    plt.title(f"Distribution of {feature_name}")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(path + f"/bias_{feature_name.replace('/', '-')}.png")
    plt.show()


def analyze_feature_importance(model, feature_names):
    feature_importance = pd.DataFrame({"feature": feature_names, "weight": model.weights})
    feature_importance["abs_weight"] = np.abs(feature_importance["weight"])
    feature_importance = feature_importance.sort_values("abs_weight", ascending=False)
    return feature_importance


def visualize_feature_importances(feature_importance, df, feature_prefix, output_path, top_n=10):
    # Filter for specific feature
    feature_data = feature_importance[feature_importance["feature"].str.startswith(feature_prefix)]

    if feature_data.empty:
        print(f"No features found with prefix '{feature_prefix}'")
        return

    # For categorical features (one-hot encoded)
    if len(feature_data) > 1:
        # Calculate distribution
        feature_distribution = df[[col for col in df.columns if col.startswith(feature_prefix)]].sum()
        feature_distribution = feature_distribution.reset_index()
        feature_distribution.columns = ["feature", "count"]

        # Merge importance and distribution
        feature_data = pd.merge(feature_data, feature_distribution, on="feature")

        # Calculate percentage
        total_count = feature_data["count"].sum()
        feature_data["percentage"] = feature_data["count"] / total_count * 100

        # Extract category names
        feature_data["category"] = feature_data["feature"].str.replace(feature_prefix, "")

        # Sort by absolute weight and get top N
        feature_data = feature_data.sort_values("abs_weight", ascending=False).head(top_n)

        # Plotting
        plt.figure(figsize=(12, 8))
        colors = ["red" if x < 0 else "green" for x in feature_data["weight"]]
        ax = sns.barplot(x="weight", y="category", data=feature_data, hue="weight", palette=colors, legend=False)

        # Add percentage labels
        for i, v in enumerate(feature_data["weight"]):
            ax.text(v, i, f' {feature_data["percentage"].iloc[i]:.1f}%', va="center")

    # For numerical features
    else:
        plt.figure(figsize=(8, 6))
        weight = feature_data["weight"].values[0]
        color = "red" if weight < 0 else "green"
        ax = plt.barh(y=[feature_prefix], width=[weight], color=color)
        plt.xlim(-max(abs(weight) * 1.1, 0.1), max(abs(weight) * 1.1, 0.1))  # Ensure some width for the bar

        # Add weight label
        plt.text(weight, 0, f" {weight:.4f}", va="center")

    plt.title(f"Feature Importance: {feature_prefix}")
    plt.xlabel("Weight (Negative = Less likely for parole, Positive = More likely)")
    plt.ylabel("Feature Category" if len(feature_data) > 1 else "Feature")
    plt.axvline(x=0, color="black", linestyle="--")
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"{feature_prefix.replace('/', '_').strip('_')}_importance.png"))
    plt.close()
