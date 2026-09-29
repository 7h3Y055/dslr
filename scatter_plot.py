import os
import sys
import matplotlib.pyplot as plt
import pandas as pd


def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/dataset_train.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: file '{dataset_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(dataset_path)

    f1, f2 = "Astronomy", "Defense Against the Dark Arts"

    houses = {
        "Gryffindor": "#ae0001",
        "Hufflepuff": "#ecb939",
        "Ravenclaw": "#222f5b",
        "Slytherin": "#2a623d"
    }

    plt.figure(figsize=(9, 6))

    for house, color in houses.items():
        house_data = df[df["Hogwarts House"] == house]
        plt.scatter(
            house_data[f1],
            house_data[f2],
            alpha=0.6,
            color=color,
            label=house,
            edgecolors="none",
            s=25
        )

    plt.title(f"Scatter Plot: {f1} vs {f2} (Similar / Collinear Features: r = -1.0)", fontsize=13, fontweight="bold")
    plt.xlabel(f1, fontsize=11)
    plt.ylabel(f2, fontsize=11)
    plt.legend(title="Hogwarts House", loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()