# DSLR - Data Science × Logistic Regression

A 42 School data science project aimed at introducing the fundamentals of data analysis, data visualization, and machine learning by classifying Hogwarts students into their respective houses using Logistic Regression.

---

## 📌 Project Overview

The project consists of three main parts:
1. **Data Analysis (`describe.py`)**: Recreating a statistical summary tool from scratch (mimicking `pandas.describe()`).
2. **Data Visualization (`histogram.py`, `scatter_plot.py`, `pair_plot.py`)**: Visualizing distributions and relationships across courses to select features.
3. **Logistic Regression (`logreg_train.py`, `logreg_predict.py`)**: Implementing a One-vs-All logistic regression model with gradient descent to predict Hogwarts houses.

---

## 🛠️ Requirements & Setup

### 1. Clone the repository
```bash
git clone https://github.com/7h3Y055/dslr.git
cd dslr
```

### 2. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Part 1: Data Analysis (`describe.py`)

`describe.py` takes a CSV dataset and calculates key descriptive statistics for all numerical features from scratch without using pandas built-in statistical functions (`mean`, `std`, `min`, `max`, `quantile`):

### Descriptive Statistics Explained

| Statistic | Description | Formula / Implementation |
| :--- | :--- | :--- |
| **Count** | The total number of valid, non-null (`non-NaN`) observations for a feature. | $N = \sum_{i=1}^n [x_i \neq \text{NaN}]$ |
| **Mean** | The arithmetic average of all non-null values, representing the central tendency of the distribution. | $\mu = \frac{1}{N} \sum_{i=1}^N x_i$ |
| **Std** | Sample standard deviation, quantifying the amount of variation or dispersion of values around the mean ($N - 1$ degrees of freedom / Bessel's correction). | $s = \sqrt{\frac{1}{N - 1} \sum_{i=1}^N (x_i - \mu)^2}$ |
| **Min** | The lowest observed value in the feature dataset. | $\min(X)$ |
| **25% (Q1)** | The first quartile; $25\%$ of observations fall below this value. Computed using linear interpolation between closest ranks. | $Q_1 = \text{Percentile}(X, 0.25)$ |
| **50% (Q2)** | The median; the midpoint value dividing the sorted dataset into two equal halves ($50\%$ below and $50\%$ above). | $Q_2 = \text{Percentile}(X, 0.50)$ |
| **75% (Q3)** | The third quartile; $75\%$ of observations fall below this value. | $Q_3 = \text{Percentile}(X, 0.75)$ |
| **Max** | The largest observed value in the feature dataset. | $\max(X)$ |
| **Skewness** | Measures the asymmetry of the probability distribution about its mean. <br>• **0**: Symmetrical distribution<br>• **> 0**: Right-skewed / positive tail<br>• **< 0**: Left-skewed / negative tail | Unbiased sample skewness:<br>$g_1 = \frac{N}{(N - 1)(N - 2)} \sum_{i=1}^N \left(\frac{x_i - \mu}{s}\right)^3$ |
| **Kurtosis** | Measures the "tailedness" of the distribution relative to a normal distribution (sample excess kurtosis). <br>• **0**: Mesokurtic (normal-like tails)<br>• **> 0**: Leptokurtic (heavy tails, higher outlier propensity)<br>• **< 0**: Platykurtic (light tails, fewer outliers) | Unbiased sample excess kurtosis:<br>$g_2 = \frac{N(N + 1)}{(N - 1)(N - 2)(N - 3)} \sum_{i=1}^N \left(\frac{x_i - \mu}{s}\right)^4 - \frac{3(N - 1)^2}{(N - 2)(N - 3)}$ |

### Usage
```bash
python describe.py
```

---

## 📁 Repository Structure

```
├── datasets/
│   ├── dataset_train.csv     # Training dataset
│   └── dataset_test.csv      # Test dataset
├── describe.py               # Statistical description program
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignored files & environments
└── README.md                 # Project documentation
```
