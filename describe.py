import os
import sys
import pandas as pd
from tabulate import tabulate


def is_valid_num(x):
    try:
        val = float(x)
        return val == val  # Returns False for NaN
    except (ValueError, TypeError):
        return False


def count(ds, col):
    n = 0.0
    for i in ds[col]:
        if is_valid_num(i):
            n += 1.0
    return n


def mean(ds, col):
    s = 0.0
    for i in ds[col]:
        if is_valid_num(i):
            s += float(i)
    c = count(ds, col)
    return s / c if c != 0 else float("nan")


def std(ds, col):
    m = mean(ds, col)
    s = 0.0
    for i in ds[col]:
        if is_valid_num(i):
            s += (float(i) - m) ** 2
    c = count(ds, col)
    return (s / (c - 1)) ** 0.5 if c > 1 else float("nan")


def minimum(ds, col):
    m = float("inf")
    for i in ds[col]:
        if is_valid_num(i):
            val = float(i)
            if val < m:
                m = val
    return m


def maximum(ds, col):
    m = float("-inf")
    for i in ds[col]:
        if is_valid_num(i):
            val = float(i)
            if val > m:
                m = val
    return m


def percentile(ds, col, p):
    values = sorted([float(x) for x in ds[col] if is_valid_num(x)])
    if not values:
        return float("nan")
    k = (len(values) - 1) * p
    f = int(k)
    c = k - f
    if f + 1 < len(values):
        return values[f] + c * (values[f + 1] - values[f])
    return values[f]


def skewness(ds, col):
    n = count(ds, col)
    if n < 3:
        return float("nan")
    m = mean(ds, col)
    s = std(ds, col)
    if s == 0 or s != s:
        return float("nan")
    sum_cubed = 0.0
    for i in ds[col]:
        if is_valid_num(i):
            sum_cubed += ((float(i) - m) / s) ** 3
    return (n / ((n - 1) * (n - 2))) * sum_cubed


def kurtosis(ds, col):
    n = count(ds, col)
    if n < 4:
        return float("nan")
    m = mean(ds, col)
    s = std(ds, col)
    if s == 0 or s != s:
        return float("nan")
    sum_fourth = 0.0
    for i in ds[col]:
        if is_valid_num(i):
            sum_fourth += ((float(i) - m) / s) ** 4
    term1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * sum_fourth
    term2 = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return term1 - term2


def main():
    if len(sys.argv) < 2:
        print("Usage: python describe.py <path_to_dataset.csv>", file=sys.stderr)
        sys.exit(1)

    dataset_path = sys.argv[1]
    if not os.path.exists(dataset_path):
        print(f"Error: file '{dataset_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        ds = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    num_cols = [col for col in ds.columns if pd.api.types.is_numeric_dtype(ds[col])]
    if not num_cols:
        print("No numerical features found in dataset.", file=sys.stderr)
        sys.exit(1)

    headers = [""] + num_cols
    matrix = [
        ["Count"] + [count(ds, col) for col in num_cols],
        ["Mean"] + [mean(ds, col) for col in num_cols],
        ["Std"] + [std(ds, col) for col in num_cols],
        ["Min"] + [minimum(ds, col) for col in num_cols],
        ["25%"] + [percentile(ds, col, 0.25) for col in num_cols],
        ["50%"] + [percentile(ds, col, 0.50) for col in num_cols],
        ["75%"] + [percentile(ds, col, 0.75) for col in num_cols],
        ["Max"] + [maximum(ds, col) for col in num_cols],
        ["Skew"] + [skewness(ds, col) for col in num_cols],
        ["Kurtosis"] + [kurtosis(ds, col) for col in num_cols],
    ]

    print(tabulate(matrix, headers=headers, tablefmt="plain", floatfmt=".6f"))


if __name__ == "__main__":
    main()
