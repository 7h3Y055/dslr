# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-source-for-stubs]
import pandas as pd

def main():
    dataset_path = "../../datasets/dataset_train.csv"
    df = pd.read_csv(dataset_path)

    courses = [
        "Arithmancy", "Care of Magical Creatures"
    ]

    houses = {
        "Gryffindor": "#ae0001",
        "Hufflepuff": "#ecb939",
        "Ravenclaw": "#222f5b",
        "Slytherin": "#2a623d"
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes = axes.flatten()

    for idx, course in enumerate(courses):
        for house, color in houses.items():
            house_scores = df[df["Hogwarts House"] == house][course].dropna()
            axes[idx].hist(house_scores, bins=25, alpha=0.5, color=color)

        axes[idx].set_title(course, fontsize=10, fontweight="bold")
        axes[idx].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Hogwarts Courses - Score Distributions Across Houses", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

if __name__ == "__main__":
    main()
