from pandas import read_csv, DataFrame
import numpy as np
import json


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

houses = [
    "Gryffindor", 
    "Hufflepuff", 
    "Ravenclaw", 
    "Slytherin"
]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    ds = read_csv("../datasets/dataset_train.csv")
    
    # 1. Clean & Standardize features
    data = ds[features]
    means = data.mean()
    stds = data.std()
    data = data.fillna(means)
    data = (data - means) / stds
    # 2. Add Bias column of 1s to features (X)
    X = np.c_[np.ones(len(data)), data]
    
    # 3. Target labels (y)
    y = ds["Hogwarts House"]
    # 4. Weights table (initialized with all 0.0)
    weights_columns = ["Bias"] + features
    weights = DataFrame(0.0, columns=weights_columns, index=houses)







    weights.loc["Mean"] = [0.0] + list(means)
    weights.loc["Std"]  = [1.0] + list(stds)
    weights.index.name = "House"
    weights.to_csv("dataset_train.csv")
    

    print("Model successfully saved to model.json!")


if __name__ == "__main__":
    main()