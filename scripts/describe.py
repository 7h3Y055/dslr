from pandas import read_csv
from tabulate import tabulate



def count(ds, col):
    n = 0.0
    for i in ds[col]:
        if str(i) != "nan":
            n += 1.0
    return n

def mean(ds, col):
    s = 0.0
    for i in ds[col]:
        if str(i) != "nan":
            s += float(i)
    c = count(ds, col)
    return s / c if c != 0 else float("nan")

def std(ds, col):
    m = mean(ds, col)
    s = 0.0
    for i in ds[col]:
        if str(i) != "nan":
            s += (float(i) - m) ** 2
    return (s / (count(ds, col) - 1)) ** 0.5

def minimum(ds, col):
    min = float("inf")
    for i in ds[col]:
        if str(i) != "nan":
            i = float(i)
            if i < min:
                min = i
    return min

def maximum(ds, col):
    max = float("-inf")
    for i in ds[col]:
        if str(i) != "nan":
            i = float(i)
            if i > max:
                max = i
    return max



def percentile(ds, col, p):
    values = sorted([float(x) for x in ds[col] if str(x) != "nan"])
    k = (len(values) - 1) * p
    f = int(k)
    c = k - f
    return values[f] + c * (values[f + 1] - values[f])
    



def skewness(ds, col):
    n = count(ds, col)
    m = mean(ds, col)
    s = std(ds, col)
    sum_cubed = 0.0
    for i in ds[col]:
        if str(i) != "nan":
            sum_cubed += ((float(i) - m) / s) ** 3
    return (n / ((n - 1) * (n - 2))) * sum_cubed

def kurtosis(ds, col):
    n = count(ds, col)
    m = mean(ds, col)
    s = std(ds, col)
    sum_fourth = 0.0
    for i in ds[col]:
        if str(i) != "nan":
            sum_fourth += ((float(i) - m) / s) ** 4
    term1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * sum_fourth
    term2 = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return term1 - term2



if __name__ == "__main__":
    ds = read_csv("../datasets/dataset_train.csv")
    headers = ["", "Index", "Arithmancy", "Astronomy", "Herbology", "Defense", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care", "Charms", "Flying"]
    matrix = [
        ["Count", count(ds, "Index"), count(ds, "Arithmancy"), count(ds, "Astronomy"), count(ds, "Herbology"), count(ds, "Defense Against the Dark Arts"), count(ds, "Divination"), count(ds, "Muggle Studies"), count(ds, "Ancient Runes"), count(ds, "History of Magic"), count(ds, "Transfiguration"), count(ds, "Potions"), count(ds, "Care of Magical Creatures"), count(ds, "Charms"), count(ds, "Flying")],
        ["Mean", mean(ds, "Index"), mean(ds, "Arithmancy"), mean(ds, "Astronomy"), mean(ds, "Herbology"), mean(ds, "Defense Against the Dark Arts"), mean(ds, "Divination"), mean(ds, "Muggle Studies"), mean(ds, "Ancient Runes"), mean(ds, "History of Magic"), mean(ds, "Transfiguration"), mean(ds, "Potions"), mean(ds, "Care of Magical Creatures"), mean(ds, "Charms"), mean(ds, "Flying")],
        ["Std", std(ds, "Index"), std(ds, "Arithmancy"), std(ds, "Astronomy"), std(ds, "Herbology"), std(ds, "Defense Against the Dark Arts"), std(ds, "Divination"), std(ds, "Muggle Studies"), std(ds, "Ancient Runes"), std(ds, "History of Magic"), std(ds, "Transfiguration"), std(ds, "Potions"), std(ds, "Care of Magical Creatures"), std(ds, "Charms"), std(ds, "Flying")],
        ["Min", minimum(ds, "Index"), minimum(ds, "Arithmancy"), minimum(ds, "Astronomy"), minimum(ds, "Herbology"), minimum(ds, "Defense Against the Dark Arts"), minimum(ds, "Divination"), minimum(ds, "Muggle Studies"), minimum(ds, "Ancient Runes"), minimum(ds, "History of Magic"), minimum(ds, "Transfiguration"), minimum(ds, "Potions"), minimum(ds, "Care of Magical Creatures"), minimum(ds, "Charms"), minimum(ds, "Flying")],
        ["25%", percentile(ds, "Index", 0.25), percentile(ds, "Arithmancy", 0.25), percentile(ds, "Astronomy", 0.25), percentile(ds, "Herbology", 0.25), percentile(ds, "Defense Against the Dark Arts", 0.25), percentile(ds, "Divination", 0.25), percentile(ds, "Muggle Studies", 0.25), percentile(ds, "Ancient Runes", 0.25), percentile(ds, "History of Magic", 0.25), percentile(ds, "Transfiguration", 0.25), percentile(ds, "Potions", 0.25), percentile(ds, "Care of Magical Creatures", 0.25), percentile(ds, "Charms", 0.25), percentile(ds, "Flying", 0.25)],
        ["50%", percentile(ds, "Index", 0.50), percentile(ds, "Arithmancy", 0.50), percentile(ds, "Astronomy", 0.50), percentile(ds, "Herbology", 0.50), percentile(ds, "Defense Against the Dark Arts", 0.50), percentile(ds, "Divination", 0.50), percentile(ds, "Muggle Studies", 0.50), percentile(ds, "Ancient Runes", 0.50), percentile(ds, "History of Magic", 0.50), percentile(ds, "Transfiguration", 0.50), percentile(ds, "Potions", 0.50), percentile(ds, "Care of Magical Creatures", 0.50), percentile(ds, "Charms", 0.50), percentile(ds, "Flying", 0.50)],
        ["75%", percentile(ds, "Index", 0.75), percentile(ds, "Arithmancy", 0.75), percentile(ds, "Astronomy", 0.75), percentile(ds, "Herbology", 0.75), percentile(ds, "Defense Against the Dark Arts", 0.75), percentile(ds, "Divination", 0.75), percentile(ds, "Muggle Studies", 0.75), percentile(ds, "Ancient Runes", 0.75), percentile(ds, "History of Magic", 0.75), percentile(ds, "Transfiguration", 0.75), percentile(ds, "Potions", 0.75), percentile(ds, "Care of Magical Creatures", 0.75), percentile(ds, "Charms", 0.75), percentile(ds, "Flying", 0.75)],
        ["Max", maximum(ds, "Index"), maximum(ds, "Arithmancy"), maximum(ds, "Astronomy"), maximum(ds, "Herbology"), maximum(ds, "Defense Against the Dark Arts"), maximum(ds, "Divination"), maximum(ds, "Muggle Studies"), maximum(ds, "Ancient Runes"), maximum(ds, "History of Magic"), maximum(ds, "Transfiguration"), maximum(ds, "Potions"), maximum(ds, "Care of Magical Creatures"), maximum(ds, "Charms"), maximum(ds, "Flying")],
        ["Skew", skewness(ds, "Index"), skewness(ds, "Arithmancy"), skewness(ds, "Astronomy"), skewness(ds, "Herbology"), skewness(ds, "Defense Against the Dark Arts"), skewness(ds, "Divination"), skewness(ds, "Muggle Studies"), skewness(ds, "Ancient Runes"), skewness(ds, "History of Magic"), skewness(ds, "Transfiguration"), skewness(ds, "Potions"), skewness(ds, "Care of Magical Creatures"), skewness(ds, "Charms"), skewness(ds, "Flying")],
        ["Kurtosis", kurtosis(ds, "Index"), kurtosis(ds, "Arithmancy"), kurtosis(ds, "Astronomy"), kurtosis(ds, "Herbology"), kurtosis(ds, "Defense Against the Dark Arts"), kurtosis(ds, "Divination"), kurtosis(ds, "Muggle Studies"), kurtosis(ds, "Ancient Runes"), kurtosis(ds, "History of Magic"), kurtosis(ds, "Transfiguration"), kurtosis(ds, "Potions"), kurtosis(ds, "Care of Magical Creatures"), kurtosis(ds, "Charms"), kurtosis(ds, "Flying")]
    ]
    print(tabulate(matrix, headers=headers, tablefmt="plain", floatfmt=".4f"))
