import argparse
import os
import sys
import pandas as pd
import numpy as np


# 10 selected courses
features = [
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Charms",
    "Flying"
]

# 4 target houses
houses = [
    "Gryffindor",
    "Hufflepuff",
    "Ravenclaw",
    "Slytherin"
]

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predict Hogwarts houses using trained logistic regression weights."
    )
    parser.add_argument("dataset_test", help="Path to dataset_test.csv")
    parser.add_argument("weights", help="Path to weights.csv")
    return parser.parse_args()


def validate_file(path, label):
    if not os.path.exists(path):
        print(f"Error: {label} file '{path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(path):
        print(f"Error: '{path}' is a directory, not a file.", file=sys.stderr)
        sys.exit(1)
    if not os.access(path, os.R_OK):
        print(f"Error: {label} file '{path}' is not readable.", file=sys.stderr)
        sys.exit(1)


def sigmoid(theta):
    return 1 / (1 + (np.e ** (-theta)))

def main():
    args = parse_arguments()
    test_path = args.dataset_test
    weights_path = args.weights

    validate_file(test_path, "Test dataset")
    validate_file(weights_path, "Weights")

    ds = pd.read_csv(test_path)[features]
    ws = pd.read_csv(weights_path, index_col=0)

    mean = ws.loc["Mean", features].astype(float)
    std = ws.loc["Std", features].astype(float)

    ds = ds.fillna(mean)
    ds = (ds - mean) / std

    predictions = []
    for s in ds.itertuples(): 
        prob = []
        x = np.array([1.0] + list(s[1:]), dtype=float)
        for _, w in ws.iloc[0:4].iterrows():
            theta = w[["Bias"] + features].to_numpy(dtype=float)
            z = np.dot(x, theta)
            prob.append(sigmoid(z))
        predictions.append(houses[np.argmax(prob)])

    df = pd.DataFrame({
        "Index": range(len(predictions)),
        "Hogwarts House": predictions
    })
    df.to_csv("houses.csv", index=False)
    print("Model successfully saved to houses.csv!")

if __name__ == "__main__":
    main()