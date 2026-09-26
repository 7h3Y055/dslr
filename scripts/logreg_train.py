import numpy as np
import pandas as pd
import sys

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


def sigmoid(theta):
    return 1 / (1 + (np.e ** (-theta)))


def batch_gradient_descent(w, X, y, m):
    for house in houses:
        y_binary = (y == house).astype(int)
        theta = np.zeros(X.shape[1], dtype=float)
        
        for _ in range(EPOCHS):
            p = sigmoid(X @ theta)
            grad = ((p - y_binary) @ X) / m
            theta = theta - (LEARNING_RATE * grad)
        w.loc[house] = theta


def stochatic_gradient_descent(w, X, y, m):
    for house in houses:
        y_binary = (y == house).astype(int)
        theta = np.zeros(X.shape[1], dtype=float)
        
        for _ in range(EPOCHS):
            for i in range(m):
                p = sigmoid(X[i] @ theta)
                grad = (p - y_binary[i]) * X[i]
                theta = theta - (LEARNING_RATE * grad)
        w.loc[house] = theta


def main(algo):



    data  = pd.read_csv("../datasets/dataset_train.csv")
    
    y = data["Hogwarts House"]
    data = data[features]

    mean = data.mean()
    std = data.std()

    data = data.fillna(mean)

    data = (data - mean) / std

    X = np.column_stack((np.ones((data.shape[0], 1)), data.to_numpy()))
    m = X.shape[0]

    w = pd.DataFrame(columns=["Bias"] + features)

    if algo == "BGD":
        batch_gradient_descent(w, X, y, m)
    elif algo == "SGD":
        stochatic_gradient_descent(w, X, y, m)
    else:
        print("Usage: python logreg_train.py <algorithm>")
        sys.exit(1)

    w.loc["Mean"] = [0] + mean.to_list()
    w.loc["Std"] = [1] + std.to_list()

    w.to_csv("weights.csv")
    print("Model successfully saved to weights.csv!")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main("BGD")
    elif sys.argv[1] == "--BGD":
        main("BGD")
    elif sys.argv[1] == "--SGD":
        main("SGD")
    else:
        print("Usage: python logreg_train.py <algorithm>")
        sys.exit(1)