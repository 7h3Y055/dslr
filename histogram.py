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

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, course in enumerate(courses):
        ax = axes[idx]
        for house, color in houses.items():
            house_scores = df[df["Hogwarts House"] == house][course].dropna()
            ax.hist(house_scores, bins=15, alpha=0.5, color=color, label=house)

        ax.set_title(course, fontsize=10, fontweight="bold")

        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=8)

    # Place unified legend in empty slot 13 (no overlap with any subplot)
    handles = [
        plt.Line2D([1], [1], marker="s", color="w", markerfacecolor=col, markersize=10, label=h) for h, col in houses.items()
    ]
    axes[13].axis("off")
    axes[13].legend(handles=handles, title="Hogwarts House", loc="center", fontsize=11, title_fontsize=12, frameon=True)

    # Remove remaining unused slots 14 and 15
    fig.delaxes(axes[14])
    fig.delaxes(axes[15])

    fig.suptitle(
        "Hogwarts Courses - Score Distributions Across Houses\n(Care of Magical Creatures & Arithmancy show homogeneous distributions)",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    plt.show()


if __name__ == "__main__":
    main()
