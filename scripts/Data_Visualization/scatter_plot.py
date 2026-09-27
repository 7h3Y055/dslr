# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-source-for-stubs]
import pandas as pd

def main():
    dataset_path = "../../datasets/dataset_train.csv"
    df = pd.read_csv(dataset_path)

    f1, f2 = "Astronomy", "Defense Against the Dark Arts"
    # f1, f2 = "Muggle Studies", "Ancient Runes"
    # f1, f2 = "History of Magic", "Divination"


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

    plt.title(f"Scatter Plot: {f1} vs {f2}", fontsize=13, fontweight="bold")
    plt.xlabel(f1, fontsize=11)
    plt.ylabel(f2, fontsize=11)
    plt.legend(title="Hogwarts House", loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()