import os
import sys
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-source-for-stubs]
import pandas as pd


def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "../../datasets/dataset_train.csv"
    if not os.path.exists(dataset_path) and os.path.exists("datasets/dataset_train.csv"):
        dataset_path = "datasets/dataset_train.csv"

    df = pd.read_csv(dataset_path)

    courses = [
        "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
        "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"
    ]

    houses = {
        "Gryffindor": "#ae0001",
        "Hufflepuff": "#ecb939",
        "Ravenclaw": "#222f5b",
        "Slytherin": "#2a623d"
    }

    n = len(courses)
    fig, axes = plt.subplots(n, n, figsize=(18, 18))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                for house, color in houses.items():
                    house_scores = df[df["Hogwarts House"] == house][courses[i]].dropna()
                    ax.hist(house_scores, bins=15, alpha=0.5, color=color)
            else:
                for house, color in houses.items():
                    sub = df[df["Hogwarts House"] == house]
                    ax.scatter(sub[courses[j]], sub[courses[i]], color=color, alpha=0.4, s=1.5, edgecolors="none")

            if i == n - 1:
                ax.set_xlabel(courses[j], fontsize=6, rotation=45, ha="right")
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(courses[i], fontsize=6)
            else:
                ax.set_yticks([])

            ax.tick_params(axis="both", which="both", labelsize=5)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=8, label=h)
        for h, col in houses.items()
    ]
    fig.legend(handles=handles, title="Hogwarts House", loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=9, title_fontsize=10)

    fig.suptitle("Pair Plot Matrix of Hogwarts Courses", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, bottom=0.06, left=0.07, right=0.98)
    plt.show()


if __name__ == "__main__":
    main()
