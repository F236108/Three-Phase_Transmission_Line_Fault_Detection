# Detecting Faults on Three-Phase Transmission Lines Using Machine Learning: From First-Principles Logistic Regression to Support Vector Machines

*A practical deep-dive into transmission line fault detection using custom NumPy implementations and Support Vector Machines.*

---

## 1. Introduction & The Engineering Challenge

In high-voltage electrical power grids, three-phase transmission lines are the critical arteries carrying electricity across thousands of miles. However, these lines are vulnerable to various fault conditions, including:
- **Phase-to-Ground (L-G) faults** (e.g., lightning strikes, fallen trees)
- **Phase-to-Phase (L-L) faults** (e.g., physical contact between phase conductors during storms)
- **Double Phase-to-Ground (L-L-G) faults**
- **Three-Phase Symmetrical (L-L-L / L-L-L-G) faults**

When a fault occurs, instantaneous currents surge to destructive levels, while voltages sag sharply due to the collapse of line impedance. A protective relay must detect this abnormal condition within **milliseconds** (typically 1–2 cycles of 50/60 Hz power frequency) and send a trip signal to the circuit breaker. 

In this project, we explore a fundamental machine learning and engineering question:
> **Can a linear decision boundary detect a fault from instantaneous 3-phase current and voltage measurements? If not, what is the simplest and most effective fix: expanding the feature space (e.g., quadratic features) or using non-linear kernels (e.g., RBF Support Vector Machines)?**

To rigorously understand the underlying mathematical mechanics, we implement **Logistic Regression completely from scratch using NumPy** (avoiding high-level abstractions like Scikit-Learn's estimators, scalers, or splitting utilities) and benchmark it against **Linear and Radial Basis Function (RBF) Support Vector Machines (SVMs)**.

---

## 2. Dataset and Preprocessing Pipeline

We utilized the **Electrical Fault Detection and Classification** benchmark dataset from Kaggle, simulated in MATLAB/Simulink. Each observation represents an instantaneous measurement sample containing 6 physical quantities:
- $I_a, I_b, I_c$: Instantaneous currents of phases A, B, and C (in Amperes).
- $V_a, V_b, V_c$: Instantaneous voltages of phases A, B, and C (in per-unit system).
- $y \in \{0, 1\}$: Target label where $1$ denotes a fault condition and $0$ denotes normal operation.

### Data Cleaning and Leak-Free Standardisation

1. **Uninformative Column Removal**: The raw dataset contained two empty trailing columns (`Unnamed: 7` and `Unnamed: 8`), which were dropped.
2. **Deterministic Partitioning**: The cleaned dataset (12,001 rows) was split into **60% Training** (7,200 rows), **20% Validation** (2,400 rows), and **20% Test** (2,401 rows) using a fixed random seed (`np.random.default_rng(42)`). The fault class proportion remained balanced across all splits (~45.1% – 46.1% faults).
3. **Strict Standardisation**: To prevent data leakage, the mean $\mu$ and standard deviation $\sigma$ were computed **solely from the training partition** and then applied to transform the validation and test partitions:
   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

### Feature Sets Defined

- **Set R (Raw Features, $d=6$)**: $[I_a, I_b, I_c, V_a, V_b, V_c]$
- **Set Q (Quadratic Features, $d=12$)**: $[I_a, I_b, I_c, V_a, V_b, V_c, I_a^2, I_b^2, I_c^2, V_a^2, V_b^2, V_c^2]$

*Crucial engineering detail*: For Set Q, the squared features were computed from the raw physical values **before** computing the 12-dimensional training mean and standard deviation for standardisation.

---

## 3. Exploratory Data Analysis & Why Linear Boundaries Fail

Before training any model, exploratory data analysis reveals the underlying electrical dynamics.

### Visualising Phase Currents over Time
Plotting $I_a, I_b, I_c$ across consecutive time samples with fault intervals highlighted shows that in normal conditions, currents follow smooth sinusoidal oscillations within a bounded amplitude. Once a fault strikes, the current amplitudes spike drastically and exhibit severe asymmetry.

*(Figure 1: Line currents for first 2000 rows with shaded fault intervals)*

### The Phase-Plane Geometry ($I_a$ vs $I_b$)
A scatter plot of $I_a$ vs $I_b$ reveals a revealing geometric structure:
- Normal operating points form a compact cluster centered at the origin.
- Fault points form a wide **annular ring (torus)** surrounding the normal operating cluster.

*(Figure 2: Scatter plot of Ia vs Ib showing the annular fault distribution)*

### Can a Linear Classifier Separate This?
**No.** Because AC currents alternate sign every half-cycle, instantaneous currents during a fault can be large positive, large negative, or momentarily passing through zero. Consequently, fault points encircle normal points in all directions in the $(I_a, I_b)$ plane. A straight hyperplane slicing through this space cannot isolate the central no-fault core from the surrounding fault perimeter without incurring massive misclassifications.

---

## 4. Machine Learning Models: Theory & Implementation

### 4.1 Logistic Regression from Scratch (NumPy)

We formulated the binary logistic regression model with parameters $\theta \in \mathbb{R}^{d+1}$:

1. **Hypothesis Function**:
   $$h_\theta(x) = g(\theta^\top x) = \frac{1}{1 + e^{-\theta^\top x}}$$

2. **Negative Log-Likelihood Cost with Numerical Clipping**:
   $$J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(h_\theta(x^{(i)})) + (1 - y^{(i)}) \ln(1 - h_\theta(x^{(i)})) \right]$$
   To prevent catastrophic numerical overflow ($\ln(0)$ resulting in `NaN`), probabilities were clipped to $[10^{-12}, 1 - 10^{-12}]$.

3. **Batch Gradient Descent Parameter Update**:
   $$\theta_j := \theta_j + \frac{\alpha}{m} \sum_{i=1}^m \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$

4. **Learning Rate Selection ($\alpha$)**:
   We evaluated $\alpha \in \{0.01, 0.1, 1, 10\}$ over 2,000 iterations. For both Set R and Set Q, the cost curve strictly decreased at every step with $\alpha = 1.0$ (and $\alpha = 10.0$ for Set Q), starting at the theoretical initial cost $J(\vec{0}) = \ln(2) \approx 0.693147$.

### 4.2 Support Vector Machines (Linear & RBF Kernels)

Using `sklearn.svm.SVC` on Set R:
- **Linear SVM**: Evaluated across $C \in \{0.1, 1.0, 10.0, 100.0\}$.
- **RBF SVM**: Evaluated across grid combinations of $C \in \{0.1, 1.0, 10.0, 100.0\}$ and $\gamma \in \{0.1, 1.0\}$.

---

## 5. Experimental Results & Performance Analysis

### 5.1 Validation Performance Summary

| Model / Feature Set | Hyperparameters | Validation Accuracy | Support Vectors |
|---|---|---|---|
| Logistic Regression (Set R) | $\alpha = 1.0$ | ~73.8% | N/A |
| Logistic Regression (Set Q) | $\alpha = 10.0$ | **98.7%** | N/A |
| Linear SVM (Set R) | $C = 100.0$ | ~74.5% | 4,210 / 7,200 |
| RBF Kernel SVM (Set R) | $C = 100.0, \gamma = 1.0$ | **99.6%** | 684 / 7,200 |

### 5.2 Final Unbiased Test Set Evaluation

Evaluating the four selected final models exactly once on the held-out test set ($N=2,401$):

| Model | Test Accuracy | True Negatives (TN) | False Alarms (FP) | Missed Faults (FN) | True Positives (TP) |
|---|---|---|---|---|---|
| **Logistic Regression (Set R)** | ~73.5% | 1,020 | 285 | 351 | 745 |
| **Linear SVM ($C=100$)** | ~74.2% | 1,034 | 271 | 348 | 748 |
| **Logistic Regression (Set Q)** | **98.8%** | 1,291 | 14 | 15 | 1,081 |
| **RBF SVM ($C=100, \gamma=1$)** | **99.7%** | 1,301 | 4 | 3 | 1,093 |

---

## 6. Deep Analytical Insights (Viva & Practical Relay Design)

### 6.1 Why Linear Models Fail Similarly on Set R
Both Logistic Regression (Set R) and the Linear SVM hover around 73–74% accuracy and miss over 300 faults. Both algorithms construct a linear decision boundary in the 6D raw input space. Because the fault geometry surrounds the normal state in phase space, no linear hyperplane can separate them.

### 6.2 Physical Interpretation of Learned Set Q Weights
Examining the learned parameter vector $\theta$ for Set Q:
- **Squared Current Weights ($I_a^2, I_b^2, I_c^2$)**: Substantially **positive** ($+1.8$ to $+3.2$). Higher current squared indicates elevated energy and power surge, driving $h_\theta(x) \to 1$ (Fault).
- **Squared Voltage Weights ($V_a^2, V_b^2, V_c^2$)**: Strongly **negative** ($-2.1$ to $-4.5$). Higher voltage squared signifies healthy nominal voltage, driving $h_\theta(x) \to 0$ (No Fault).
This demonstrates how feature engineering with physics domain knowledge enables simple linear models to solve complex non-linear problems.

### 6.3 Support Vector Behavior & Regularization parameter $C$
The Linear SVM required over 4,200 support vectors (more than 58% of the entire training dataset), confirming severe margin violations and non-separability. In contrast, the RBF SVM mapped the data into an infinite-dimensional reproducing kernel Hilbert space (RKHS) where clean separation was achieved with only ~680 support vectors. Increasing $C$ heavily penalizes margin slack, narrowing the margin while maximizing separation accuracy.

### 6.4 Temporal Leakage Considerations in Time-Series Splitting
Since consecutive rows in the dataset represent contiguous time steps of simulated power waveforms, random uniform row-splitting causes adjacent time steps of the same event to land in both training and test sets. In real-world deployment on physical substations, relays experience distinct transient fault waveforms across varying impedance paths and fault angles. Thus, benchmark test accuracy is slightly **optimistic** compared to field tests on unobserved transmission topologies.

### 6.5 Which Model to Deploy in a Real Protective Relay?
In protection engineering, **a missed fault (False Negative) is catastrophic**—it leads to equipment explosion, busbar destruction, and grid-wide blackout. A false alarm (False Positive) is merely an inconvenience that opens a breaker redundantly.
Therefore, the **RBF SVM (or Set Q Logistic Regression)** is the clear deployment choice:
- It reduces missed faults from >340 down to just **3**.
- Set Q Logistic Regression offers ultra-low latency ($\mathcal{O}(d)$ dot-product inference) suitable for embedded microcontrollers inside digital relays.

---

## 7. Interactive Web Application

We developed a full-stack **Streamlit** dashboard enabling engineers and operators to input 3-phase instantaneous currents ($I_a, I_b, I_c$) and voltages ($V_a, V_b, V_c$) and view real-time fault predictions from both the Quadratic Logistic Regression and RBF SVM models side by side.

```bash
streamlit run app/app.py
```

---

## 8. Conclusion & Key Takeaways

1. **Raw instantaneous AC measurements are non-linearly separable** due to sinusoidal symmetry.
2. **Domain-informed feature engineering** ($I^2$ and $V^2$) lifts linear logistic regression from 73% to >98.8% accuracy.
3. **Kernel SVMs** achieve state-of-the-art 99.7% accuracy by transforming the geometric representation directly.
4. Implementing algorithms from scratch provides unmatched clarity into the numerical mechanics of gradient descent and boundary formation.

---
*Code Repository*: [GitHub Repository Link]*  
*Interactive Demo*: Streamlit Web App in `app/app.py`
