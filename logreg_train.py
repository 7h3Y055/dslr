import os
import sys
import numpy as np
import pandas as pd

# Hyperparameters
EPOCHS = 1000
LEARNING_RATE = 0.1

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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def compute_mean_std(df, feature_cols):
    """Compute mean and std"""
    means = {}
    stds = {}
    for col in feature_cols:
        vals = [float(x) for x in df[col] if float(x) == float(x)]
        n = len(vals)
        if n <= 1:
            print(f"Error: Not enough data for feature '{col}' to compute mean and std.")
            return None
        m = sum(vals) / n
        variance = sum((x - m) ** 2 for x in vals) / (n - 1)
        s = variance ** 0.5
        means[col] = m
        stds[col] = s
    return pd.Series(means), pd.Series(stds)


def batch_gradient_descent(w, X, y, m):
    for house in houses:
        y_binary = (y == house).astype(int)
        theta = np.zeros(X.shape[1], dtype=float)

        for _ in range(EPOCHS):
            p = sigmoid(X @ theta)
            grad = ((p - y_binary) @ X) / m
            theta = theta - (LEARNING_RATE * grad)
        w.loc[house] = theta


def stochastic_gradient_descent(w, X, y, m):
    for house in houses:
        y_binary = (y == house).astype(int)
        theta = np.zeros(X.shape[1], dtype=float)

        for _ in range(EPOCHS):
            for i in range(m):
                p = sigmoid(np.dot(X[i], theta))
                grad = (p - y_binary[i]) * X[i]
                theta = theta - (LEARNING_RATE * grad)
        w.loc[house] = theta


def parse_arguments():
    dataset_path = "datasets/dataset_train.csv"
    algo = "BGD"

    args = sys.argv[1:]
    for arg in args:
        if arg in ("--BGD", "-BGD"):
            algo = "BGD"
        elif arg in ("--SGD", "-SGD"):
            algo = "SGD"
        elif not arg.startswith("-"):
            dataset_path = arg
        else:
            print(f"Unknown option: {arg}", file=sys.stderr)
            print("Usage: python logreg_train.py <dataset_train.csv> [--BGD | --SGD]", file=sys.stderr)
            sys.exit(1)

    if not os.path.exists(dataset_path):
        print(f"Error: file '{dataset_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    return dataset_path, algo


def main():
    dataset_path, algo = parse_arguments()

    data = pd.read_csv(dataset_path)
    y = data["Hogwarts House"]

    # Calculate mean and standard deviation without forbidden built-in functions
    mean, std = compute_mean_std(data, features)

    X_data = data[features].fillna(mean)
    X_data = (X_data - mean) / std

    X = np.column_stack((np.ones((X_data.shape[0], 1)), X_data.to_numpy()))
    m = X.shape[0]

    w = pd.DataFrame(columns=["Bias"] + features)

    if algo == "BGD":
        batch_gradient_descent(w, X, y, m)
    elif algo == "SGD":
        stochastic_gradient_descent(w, X, y, m)

    w.loc["Mean"] = [0] + mean.to_list()
    w.loc["Std"] = [1] + std.to_list()

    w.to_csv("weights.csv")
    print(f"Model successfully trained using {algo} and saved to weights.csv!")


if __name__ == "__main__":
    main()