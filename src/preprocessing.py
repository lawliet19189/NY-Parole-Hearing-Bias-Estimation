from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np


def parse_time_period(time_str):
    """Parse a time period string like '01-06' into a number of months."""
    if pd.isna(time_str) or time_str == "None":
        return np.nan
    try:
        years, months = map(int, time_str.split("-"))
        return years * 12 + months
    except:
        return np.nan


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """This function preprocesses the data by converting categorical variables to numerical and dates to days since a reference date."""

    # convert categorical variables to numerical
    data["Interview Decision"] = (data["Interview Decision"] == "PAROLE").astype(int)

    # data["Interview Decision"] = data["Interview Decision"].map(
    #     {
    #         "GRANTED": 1,
    #         "OPEN DT": 1,
    #         "PAROLED": 1,
    #         "OR EARLIER": 1,
    #         "OPEN DATE": 1,
    #         "*": 0,
    #         "**********": 0,
    #         "OTHER": 0,
    #         "RCND&HOLD": 0,
    #         "REINSTATE": 0,
    #         "RCND&RELSE": 0,
    #     }
    # )

    date_columns = [
        "Birth Date",
        "Parole Board Interview Date",
        "Release Date",
        "Parole Eligibility Date",
        "Conditional Release Date",
        "Maximum Expiration Date",
        "Parole ME Date",
        "Post Release Supervision ME Date",
        "Parole Board Discharge Date",
    ]

    # Convert dates to days since a reference date
    reference_date = datetime(1970, 1, 1)

    # minor hack to cases when the value is not date but life sentence
    lifesentence_data = (reference_date + pd.Timedelta(days=36525)).strftime("%m/%d/%Y")

    # convert date columns to datetime
    for col in date_columns:
        if col in data.columns:
            data[col] = data[col].apply(
                lambda x: np.nan if x in (None, "NONE", "None", "N/A", "n/a", "NA", "na", "NaN", "nan") else x
            )

            data[col] = data[col].apply(lambda x: lifesentence_data if x == "LIFE SENTENCE" else x)

            data[col] = (pd.to_datetime(data[col]) - reference_date).dt.days

    # one hot encoding of categorical variables
    categorical_columns = [
        "Race/ethnicity",
        "Housing or Interview Facility",
        "Parole Board Interview Type",
        "Housing/Release Facility",
    ]

    # iterate over the categorical columns and create dummies for each category
    # delete the original column
    for col in categorical_columns:
        if col in data.columns:
            dummies = pd.get_dummies(data[col], prefix=col, dummy_na=True)
            data = pd.concat([data, dummies], axis=1)
            data.drop(col, axis=1, inplace=True)

    return data


def process_crimes(crime_string) -> dict[str, int]:
    """Process the crimes string and extract relevant information."""
    crimes = crime_string.split(";")
    crime_types = []
    crime_classes = []
    crime_places = []

    for crime in crimes:
        if not crime.strip():
            continue
        parts = crime.strip().split("(")
        if len(parts) == 2:
            crime_type = parts[0].strip()
            class_and_place = parts[1].strip(")").split(",")
            if len(class_and_place) == 2:
                crime_class = class_and_place[0].strip().split()[-1]
                crime_place = class_and_place[1].strip()

                crime_types.append(crime_type)
                crime_classes.append(crime_class)
                crime_places.append(crime_place)

    return {
        "crime_count": len(crime_types),
        "crime_types": Counter(crime_types),
        "crime_classes": Counter(crime_classes),
        "crime_places": Counter(crime_places),
    }


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features from the data."""

    # The crimes column contains a list of crimes committed by the parolee in a string format.
    crime_data = data["Crimes"].apply(process_crimes)

    # New feature for the number of crimes committed
    data["Crime Count"] = crime_data.apply(lambda x: x["crime_count"])

    # Get top N most common crime types, classes, and places
    N = 50
    top_crime_types = set()
    top_crime_classes = set()
    top_crime_places = set()

    for crime_info in crime_data:
        top_crime_types.update([crime for crime, _ in crime_info["crime_types"].most_common(N)])
        top_crime_classes.update([class_ for class_, _ in crime_info["crime_classes"].most_common(N)])
        top_crime_places.update([place for place, _ in crime_info["crime_places"].most_common(N)])

    # Create binary features for top crime types, classes, and places
    new_features = {}

    for crime_type in top_crime_types:
        new_features[f"Crime_Type_{crime_type}"] = crime_data.apply(lambda x: int(crime_type in x["crime_types"]))

    for crime_class in top_crime_classes:
        new_features[f"Crime_Class_{crime_class}"] = crime_data.apply(lambda x: int(crime_class in x["crime_classes"]))

    for crime_place in top_crime_places:
        new_features[f"Crime_Place_{crime_place}"] = crime_data.apply(lambda x: int(crime_place in x["crime_places"]))

    new_features_df = pd.DataFrame(new_features)

    data = pd.concat([data, new_features_df], axis=1)

    # Convert Aggregate Minimum/Maximum Sentence to months
    data["Aggregated Minimum Sentence Months"] = data["Aggregated Minimum Sentence"].apply(parse_time_period)
    data["Aggregated Maximum Sentence Months"] = data["Aggregated Maximum Sentence"].apply(parse_time_period)

    # Calculate Time Served and Remaining Sentence
    data["Time Served Months"] = data["Aggregated Minimum Sentence Months"]
    data["Remaining Sentence Months"] = (
        data["Aggregated Maximum Sentence Months"] - data["Aggregated Minimum Sentence Months"]
    )

    # Handle cases where Remaining Sentence is negative (due to data issues)
    data.loc[data["Remaining Sentence Months"] < 0, "Remaining Sentence Months"] = 0

    # Calculate age at interview (in years)
    if "Birth Date_diff" in data.columns:
        data["Age at Interview"] = data["Birth Date_diff"] / -365.25

    # Drop unnecessary columns
    columns_to_drop = ["Name", "DIN", "Aggregated Minimum Sentence", "Aggregated Maximum Sentence", "Crimes"]
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    return data
