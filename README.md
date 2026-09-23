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

| Statistic    | Description                                                                                                                                                                                                                                                                      | Formula / Implementation                                                                                                                                               |
| :----------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Count**    | The total number of valid, non-null (`non-NaN`) observations for a feature.                                                                                                                                                                                                      | $N = \sum_{i=1}^n [x_i \neq \text{NaN}]$                                                                                                                               |
| **Mean**     | The arithmetic average of all non-null values, representing the central tendency of the distribution.                                                                                                                                                                            | $\mu = \frac{1}{N} \sum_{i=1}^N x_i$                                                                                                                                   |
| **Std**      | Sample standard deviation, quantifying the amount of variation or dispersion of values around the mean ($N - 1$ degrees of freedom / Bessel's correction).                                                                                                                       | $s = \sqrt{\frac{1}{N - 1} \sum_{i=1}^N (x_i - \mu)^2}$                                                                                                                |
| **Min**      | The lowest observed value in the feature dataset.                                                                                                                                                                                                                                | $\min(X)$                                                                                                                                                              |
| **25% (Q1)** | The first quartile; $25\%$ of observations fall below this value. Computed using linear interpolation between closest ranks.                                                                                                                                                     | $Q_1 = \text{Percentile}(X, 0.25)$                                                                                                                                     |
| **50% (Q2)** | The median; the midpoint value dividing the sorted dataset into two equal halves ($50\%$ below and $50\%$ above).                                                                                                                                                                | $Q_2 = \text{Percentile}(X, 0.50)$                                                                                                                                     |
| **75% (Q3)** | The third quartile; $75\%$ of observations fall below this value.                                                                                                                                                                                                                | $Q_3 = \text{Percentile}(X, 0.75)$                                                                                                                                     |
| **Max**      | The largest observed value in the feature dataset.                                                                                                                                                                                                                               | $\max(X)$                                                                                                                                                              |
| **Skewness** | Measures the asymmetry of the probability distribution about its mean. <br>• **0**: Symmetrical distribution<br>• **> 0**: Right-skewed / positive tail<br>• **< 0**: Left-skewed / negative tail                                                                                | Unbiased sample skewness:<br>$g_1 = \frac{N}{(N - 1)(N - 2)} \sum_{i=1}^N \left(\frac{x_i - \mu}{s}\right)^3$                                                          |
| **Kurtosis** | Measures the "tailedness" of the distribution relative to a normal distribution (sample excess kurtosis). <br>• **0**: Mesokurtic (normal-like tails)<br>• **> 0**: Leptokurtic (heavy tails, higher outlier propensity)<br>• **< 0**: Platykurtic (light tails, fewer outliers) | Unbiased sample excess kurtosis:<br>$g_2 = \frac{N(N + 1)}{(N - 1)(N - 2)(N - 3)} \sum_{i=1}^N \left(\frac{x_i - \mu}{s}\right)^4 - \frac{3(N - 1)^2}{(N - 2)(N - 3)}$ |

### Usage

```bash
python scripts/describe.py
```

---

## 📈 Part 2: Data Visualization

### Histogram (`Histogram.py`)

**Question**: _Which Hogwarts course has a homogeneous score distribution between all four houses?_

**Answer**: **Care of Magical Creatures** and **Arithmancy**

- **Care of Magical Creatures**: Has virtually identical normal distributions across all 4 houses (mean $\approx 0$, std $\approx 1$, fully overlapping bell curves).
- **Arithmancy**: Also shows almost indistinguishable distributions across all 4 houses (means around $49,000$ to $50,000$, standard deviations around $15,000$ to $19,000$).
- Because their distributions overlap completely across all houses, both features provide no discriminatory separation power for house classification and should be discarded during feature selection for logistic regression.

#### Usage

```bash
python scripts/Data_Visualization/Histogram.py
```

Or specify a custom dataset path:

```bash
python scripts/Data_Visualization/Histogram.py datasets/dataset_train.csv
```

### Scatter Plot (`Scatter-plot.py`)

**Question**: _What are the two features that are similar?_

**Answer**: **Astronomy** and **Defense Against the Dark Arts**

- They show a perfect inverse linear relationship with a Pearson correlation coefficient of $r = -1.0000$.
- In the scatter plot, every student's score in Astronomy mirrors their Defense Against the Dark Arts score along a straight negative diagonal line.
- Because one feature can be derived directly from the other, they are collinear; keeping both in a linear classifier would be redundant.

#### Usage

```bash
python scripts/Data_Visualization/Scatter-plot.py
```

---

### Pair Plot (`Pair-plot.py`)

**Question**: _From this pair plot, what features are you going to use for your logistic regression?_

**Answer**: Use the **10 features** with distinct house separation:
- **Discarded Features**:
  - **Arithmancy** & **Care of Magical Creatures**: Identical score distributions across all four houses (homogeneous distributions $\to$ zero discriminative power).
  - **Astronomy** *(or Defense Against the Dark Arts)*: Collinear ($r = -1.0000$), duplicate information.
- **Selected Features**: `Herbology`, `Defense Against the Dark Arts`, `Divination`, `Muggle Studies`, `Ancient Runes`, `History of Magic`, `Transfiguration`, `Potions`, `Charms`, and `Flying`.

#### Usage

```bash
python scripts/Data_Visualization/Pair-plot.py
```

---

## 🤖 Part 3: Logistic Regression (`logreg_train.py`, `logreg_predict.py`)

### Mathematical Derivation of the Gradient (Partial Derivative)

Logistic regression models the probability that a given student belongs to a specific house ($y = 1$) versus any other house ($y = 0$) using the **Sigmoid (logistic) function**.

#### 1. Hypothesis & Score Function

For a student with $n$ standardized features $x = [x_1, x_2, \dots, x_n]^T$ and an added bias term $x_0 = 1$:

$$z = \theta_0 x_0 + \theta_1 x_1 + \dots + \theta_n x_n = \theta^T x$$

The hypothesis function $h_\theta(x)$ computes the predicted probability $p \in (0, 1)$:

$$p = h_\theta(x) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

#### 2. Loss Function (Binary Cross-Entropy / Log Loss)

To ensure convexity for gradient descent, we use the negative log-likelihood (Binary Cross-Entropy):

For a single observation:
$$L(\theta) = - \left[ y \ln(p) + (1 - y) \ln(1 - p) \right]$$

For the entire training dataset of $m$ students:
$$J(\theta) = - \frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(p^{(i)}) + (1 - y^{(i)}) \ln(1 - p^{(i)}) \right]$$

#### 3. Step-by-Step Chain Rule Derivation

To update each parameter $\theta_j$ using gradient descent, we compute the partial derivative $\frac{\partial L}{\partial \theta_j}$. By the **Chain Rule**:

$$\frac{\partial L}{\partial \theta_j} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z} \cdot \frac{\partial z}{\partial \theta_j}$$

Let's compute each of the three factors individually:

##### Factor 1: $\frac{\partial L}{\partial p}$ (Derivative of Loss with respect to prediction $p$)
$$\begin{aligned}
\frac{\partial L}{\partial p} &= - \left[ y \cdot \frac{1}{p} + (1 - y) \cdot \frac{1}{1 - p} \cdot (-1) \right] \\
&= - \frac{y}{p} + \frac{1 - y}{1 - p} \\
&= \frac{-y(1 - p) + p(1 - y)}{p(1 - p)} \\
&= \frac{-y + yp + p - yp}{p(1 - p)} \\
&= \frac{p - y}{p(1 - p)}
\end{aligned}$$

##### Factor 2: $\frac{\partial p}{\partial z}$ (Derivative of the Sigmoid function)
Recall $p = \sigma(z) = (1 + e^{-z})^{-1}$:
$$\begin{aligned}
\frac{\partial p}{\partial z} &= - (1 + e^{-z})^{-2} \cdot (-e^{-z}) \\
&= \frac{e^{-z}}{(1 + e^{-z})^2} \\
&= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} \\
&= \frac{1}{1 + e^{-z}} \cdot \left( 1 - \frac{1}{1 + e^{-z}} \right) \\
&= p(1 - p)
\end{aligned}$$

##### Factor 3: $\frac{\partial z}{\partial \theta_j}$ (Derivative of linear score with respect to weight $\theta_j$)
$$\frac{\partial z}{\partial \theta_j} = \frac{\partial}{\partial \theta_j} \left( \sum_{k=0}^n \theta_k x_k \right) = x_j$$

##### Multiplying the Factors Together (Algebraic Cancellation)
Substituting Factors 1, 2, and 3 back into the Chain Rule:

$$\begin{aligned}
\frac{\partial L}{\partial \theta_j} &= \left( \frac{p - y}{p(1 - p)} \right) \cdot \Big( p(1 - p) \Big) \cdot x_j \\
&= (p - y) \cdot x_j
\end{aligned}$$

> Notice how the non-linear denominator $p(1 - p)$ from the loss derivative perfectly cancels the $p(1 - p)$ from the sigmoid derivative. This leaves an elegant, linear error term: **$(\text{prediction} - \text{truth}) \times \text{feature}$**.

#### 4. Total Cost Gradient & Vectorized Form

Averaging across all $m$ training examples:

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

In vectorized matrix notation:

$$\nabla_\theta J(\theta) = \frac{1}{m} X^T (\hat{y} - y)$$

Where:
- $X$ is the $(m \times 11)$ design matrix (including the bias column).
- $\hat{y} = \sigma(X\theta)$ is the $(m \times 1)$ vector of predictions.
- $y$ is the $(m \times 1)$ binary target vector.

#### 5. Gradient Descent Update Rule

For each epoch, the weights vector $\theta$ is updated in the opposite direction of the gradient:

$$\theta := \theta - \alpha \cdot \frac{1}{m} X^T (\hat{y} - y)$$

Where $\alpha$ is the learning rate (`LEARNING_RATE = 0.1`).

---

### Training Execution (`logreg_train.py`)

```bash
python scripts/logreg_train.py
```
This trains four One-vs-All classifiers and outputs the optimized weights along with the feature `Mean` and `Std` to `weights.csv`.

---

## 📁 Repository Structure

```
├── datasets/
│   ├── dataset_train.csv          # Training dataset
│   └── dataset_test.csv           # Test dataset
├── scripts/
│   ├── describe.py                # Statistical description program
│   ├── logreg_train.py            # Logistic regression training script
│   ├── logreg_predict.py          # Logistic regression prediction script
│   └── Data_Visualization/
│       ├── Histogram.py           # Feature distribution analysis
│       ├── Scatter-plot.py        # Collinearity analysis
│       └── Pair-plot.py           # Pairwise feature matrix
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignored files & environments
└── README.md                      # Project documentation
```
