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
    return 1 / (1 + np.e ** -z)


def main():
    # 1. Load data
    ds = pd.read_csv("../datasets/dataset_train.csv")

    # 2. Extract features
    data = ds[features]
    
    # Calculate means and stds for normalization
    means = data.mean()
    stds = data.std()

    # Fill NaN values with the mean of each feature
    data = data.fillna(means)
    
    # Standardize features  
    data = (data - means) / stds

    # 3. Add bias column of 1s (shape becomes: 1600 x 11)
    X = np.c_[np.ones(len(data)), data]
    y = ds["Hogwarts House"]
    m = len(X)

    # 4. Table to store the 11 weights for each house
    weights = pd.DataFrame(0.0, columns=["Bias"] + features, index=houses)

    # 5. One-vs-All Training loop
    for house in houses:
        y_binary = (y == house).astype(int)
        theta = np.zeros(11)

        for _ in range(EPOCHS):
            predictions = sigmoid(X @ theta)
            error = predictions - y_binary
            gradient = (X.T @ error) / m
            theta -= LEARNING_RATE * gradient

        weights.loc[house] = list(theta)

    # 6. Save weights + normalization data
    weights.loc["Mean"] = [0.0] + list(means)
    weights.loc["Std"] = [1.0] + list(stds)
    weights.to_csv("weights.csv")


    print("Model successfully saved to weights.csv!")


if __name__ == "__main__":
    main()