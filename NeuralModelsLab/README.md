# Neural Models Lab: Learning, Depth, Activations, and Output Layers

This laboratory investigates representation, non-linear hidden layers, backpropagation gradients, weight symmetry breaking, activation dynamics, and multiclass softmax output layers using PyTorch.

---

## 1. Learning Objectives

- Explain why non-linear hidden representations are mathematically required for XOR-like decision boundaries.
- Formulate task-dependent pairings between model output activations and loss functions.
- Inspect and verify reverse-mode automatic differentiation ($\partial \mathcal{L} / \partial W^{(1)}$).
- Demonstrate the symmetry failure caused by identical/zero weight initialization.
- Compare activation function behaviors (Sigmoid, Tanh, ReLU) under gradient descent.
- Implement a 3-class categorical decision extension with numerical stability diagnostics for Softmax.

---

## 2. Motivating Scenario: Redundant Safety Sensors

An autonomous system monitors two redundant binary sensors ($x_1, x_2$). A warning $y=1$ is triggered if and only if exactly one sensor is active (sensor disagreement):

| $x_1$ | $x_2$ | Meaning | Disagreement Warning ($y$) |
| :---: | :---: | :--- | :---: |
| 0 | 0 | Both inactive | 0 |
| 0 | 1 | Disagreement | 1 |
| 1 | 0 | Disagreement | 1 |
| 1 | 1 | Both active | 0 |

---

## 3. Task 1: Problem Specification & Linear Inseparability

- **Input Space**: $\mathcal{X} = \{0, 1\}^2 \subset \mathbb{R}^2$
- **Output Space**: $\mathcal{Y} = \{0, 1\}$
- **Dataset**: $\mathcal{D} = \{ ((0,0), 0), ((0,1), 1), ((1,0), 1), ((1,1), 0) \}$

### Geometric Inseparability Proof
In the 2D plane:
- Class $1$ points are $(0, 1)$ and $(1, 0)$. The line connecting them has midpoint $(0.5, 0.5)$.
- Class $0$ points are $(0, 0)$ and $(1, 1)$. The line connecting them also has midpoint $(0.5, 0.5)$.

Because the convex hulls of Class $0$ and Class $1$ intersect at $(0.5, 0.5)$, **no straight hyper-plane $w_1 x_1 + w_2 x_2 + b = 0$ can separate the two classes**.

### Linear Baseline Experiment
A single affine layer followed by Sigmoid was trained for 2,000 steps (`LinearBaseline`):
- **Final BCE Loss**: `0.6931` ($-\ln(0.5)$)
- **Predictions**: `[0.5000, 0.5000, 0.5000, 0.5000]`
- **Observation**: The linear model collapses to predicting the unconditional base rate (50% probability), achieving at most $50\%$ to $75\%$ accuracy.

---

## 4. Task 2: Model Design & Validation Criteria

### Architecture: 2-2-1 MLP
$$\boldsymbol{h} = \sigma\left( W^{(1)} \boldsymbol{x} + \boldsymbol{b}^{(1)} \right), \quad z = W^{(2)} \boldsymbol{h} + b^{(2)}, \quad \hat{y} = \sigma(z)$$
Where $W^{(1)} \in \mathbb{R}^{2 \times 2}$, $\boldsymbol{b}^{(1)} \in \mathbb{R}^2$, $W^{(2)} \in \mathbb{R}^{1 \times 2}$, $b^{(2)} \in \mathbb{R}$.

1. **Why is the hidden nonlinearity necessary?**
   Composing linear transformations yields another linear transformation: $W^{(2)}(W^{(1)} \boldsymbol{x} + \boldsymbol{b}^{(1)}) + b^{(2)} = \tilde{W}\boldsymbol{x} + \tilde{b}$. Adding depth without non-linear activations cannot increase representational expressivity. Non-linear activations warp the coordinate space so the four points become linearly separable in the hidden unit space $\boldsymbol{h}$.
2. **Why Sigmoid + Binary Cross-Entropy?**
   Interpreting logits via the logistic link $\sigma(z) = \frac{1}{1 + e^{-z}}$ corresponds to the Bernoulli likelihood. Combining them into `BCEWithLogitsLoss` yields the log-loss:
   $$\mathcal{L} = -y \ln \sigma(z) - (1-y) \ln(1 - \sigma(z))$$
   Its gradient with respect to logit $z$ is simply $\hat{y} - y$, providing strong linear gradients when predictions are wrong and avoiding numerical underflow.
3. **Validation Criteria**:
   - Final loss $< 0.01$.
   - All 4 predictions matching ground truth labels ($4/4$ correct).
   - Non-zero gradient norms on first-layer parameters.

---

## 5. Task 3 & 4: Implementation and Experimental Results

Implementation file: [`neural_models.py`](file:///c:/Users/sarthak/Desktop/i/Acads-coding/AI_Labs/NeuralModelsLab/neural_models.py).

### How to Run
```bash
python NeuralModelsLab/neural_models.py
```

### Part A: Basic Learning Check
- **Initial Loss**: `0.7482`
- **Final Loss**: `0.000772`
- **Predictions**:

| Input ($x_1, x_2$) | Target ($y$) | Model Probability | Prediction ($\hat{y}$) |
| :---: | :---: | :---: | :---: |
| $(0, 0)$ | 0 | `0.0007` | **0** |
| $(0, 1)$ | 1 | `0.9993` | **1** |
| $(1, 0)$ | 1 | `0.9990` | **1** |
| $(1, 1)$ | 0 | `0.0007` | **0** |

All four samples are classified with $>99.9\%$ confidence.

---

### Part B: Backpropagation Gradient Check
Inspecting `model.fc1.weight.grad`:
```text
tensor([[ 1.2895e-05, -1.4429e-05],
        [ 6.8115e-06, -9.7517e-06]])
```
- **Interpretation**: `parameter.grad` contains the exact analytical derivative $\frac{\partial \mathcal{L}}{\partial W^{(1)}}$ obtained via reverse-mode automatic differentiation.
- **Why is it the average?** Because PyTorch's `criterion` defaults to `reduction='mean'`, the total loss is $\mathcal{L} = \frac{1}{4} \sum_{i=1}^4 \mathcal{L}_i$, and by linearity of differentiation:
  $$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \frac{1}{4} \sum_{i=1}^4 \frac{\partial \mathcal{L}_i}{\partial W^{(1)}}$$

---

### Part C: Zero-Weight Initialization & Symmetry Failure
When all weights and biases are initialized to $0.0$:

```text
Step   1 | Unit 1: [0.0, 0.0] | Unit 2: [0.0, 0.0] | Identical: True
Step   5 | Unit 1: [0.0, 0.0] | Unit 2: [0.0, 0.0] | Identical: True
Step  20 | Unit 1: [0.0, 0.0] | Unit 2: [0.0, 0.0] | Identical: True
Step 100 | Unit 1: [0.0, 0.0] | Unit 2: [0.0, 0.0] | Identical: True
Step 200 | Unit 1: [0.0, 0.0] | Unit 2: [0.0, 0.0] | Identical: True
Final Loss: 0.6931 (Failed to learn)
```

- **Explanation**: If two hidden neurons receive identical inputs, have identical weights, and share symmetric outgoing weights, they compute identical activations ($h_1 = h_2$) and receive identical backpropagation gradients ($\frac{\partial \mathcal{L}}{\partial W_{1,:}} = \frac{\partial \mathcal{L}}{\partial W_{2,:}}$). They update identically at every step, collapsing the 2-hidden-unit network into an effective single-unit network that cannot solve XOR. Random initialization is essential to **break symmetry**.

---

### Part D: Activation Function Benchmark

| Hidden Activation | Final Loss | 4/4 Correct? | Early $\|\nabla_{W^{(1)}} \mathcal{L}\|_2$ | Behavior Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Sigmoid** | `0.003467` | **True** | `0.011808` | Smooth, monotonic, converges cleanly. |
| **Tanh** | `0.000721` | **True** | `0.033798` | Zero-centered outputs give faster early gradients. |
| **ReLU** | `0.346697` | **False** | `0.052575` | Sits at local saddle point when pre-activations are $\le 0$. |

#### Dying ReLU vs. Saturated Sigmoid ("Think About It")
- **Saturated Sigmoid**: For large positive or negative pre-activation $a$, $\sigma(a)(1 - \sigma(a)) \to 0$. The derivative is small but non-zero ($>0$), allowing slow gradient leakage.
- **Negative ReLU**: For pre-activation $a < 0$, $\text{ReLU}'(a) \equiv 0.0$. The gradient is identically zero ("dead neuron"). If both hidden units in a 2-2-1 net deactivate on negative pre-activations, all learning signals permanently vanish.
- **How to distinguish**: Inspect pre-activations $a$ before the activation function. If $|a| \gg 0$ with Sigmoid, it is saturated; if $a < 0$ with ReLU, it is dead.

---

## 6. Task 5: Three-Class Sensor Extension

Classes:
- `0`: Both inactive $(0, 0)$
- `1`: Disagree $(0, 1)$ or $(1, 0)$
- `2`: Both active $(1, 1)$

### Theoretical Predictions:
1. **Shape of final weight matrix**: $W^{(2)} \in \mathbb{R}^{3 \times 4}$ (3 classes, 4 hidden units).
2. **Number of logits per example**: 3 logits $[z_0, z_1, z_2]$.
3. **Why Softmax probabilities sum to 1**:
   $$p_k = \frac{e^{z_k}}{\sum_{j=1}^3 e^{z_j}} \implies \sum_{k=1}^3 p_k = \frac{\sum_{k=1}^3 e^{z_k}}{\sum_{j=1}^3 e^{z_j}} = 1$$
4. **Why the logit gradient is $p - y$**:
   Let loss $\mathcal{L} = -\sum_k y_k \ln p_k$ where $\sum_k y_k = 1$.
   Using $\frac{\partial p_k}{\partial z_i} = p_k(\delta_{ki} - p_i)$:
   $$\frac{\partial \mathcal{L}}{\partial z_i} = -\sum_k \frac{y_k}{p_k} \frac{\partial p_k}{\partial z_i} = -\frac{y_i}{p_i} p_i(1 - p_i) - \sum_{k \ne i} \frac{y_k}{p_k}(-p_k p_i) = -y_i(1 - p_i) + p_i \sum_{k \ne i} y_k$$
   Since $\sum_{k \ne i} y_k = 1 - y_i$:
   $$\frac{\partial \mathcal{L}}{\partial z_i} = -y_i + y_i p_i + p_i - p_i y_i = p_i - y_i \quad \iff \quad \nabla_{\boldsymbol{z}} \mathcal{L} = \boldsymbol{p} - \boldsymbol{y}$$

### Experimental Outputs:
- **Final Cross-Entropy Loss**: `0.000037`
- **Predicted Class Probabilities**:
  - `(0, 0)` $\to$ `[1.0000, 0.0000, 0.0000]` (Class 0)
  - `(0, 1)` $\to$ `[0.0000, 1.0000, 0.0000]` (Class 1)
  - `(1, 0)` $\to$ `[0.0000, 1.0000, 0.0000]` (Class 1)
  - `(1, 1)` $\to$ `[0.0000, 0.0000, 1.0000]` (Class 2)

### Softmax Numerical Stability Diagnostic
We added a constant $c = +100$ to all logits: $\boldsymbol{z}' = \boldsymbol{z} + 100$.
- **Max Absolute Difference**: `1.67e-10` ($\approx 0$)
- **Proof of Shift Invariance**:
  $$\frac{e^{z_k + c}}{\sum_j e^{z_j + c}} = \frac{e^c e^{z_k}}{e^c \sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}}$$
- **Engineering Significance**: To prevent numerical overflow (`inf` from `exp(z)` for large positive logits), robust libraries (including PyTorch) subtract $\max_j z_j$ before exponentiating.

---

## 7. Reflection Questions

1. **Difference between depth and nonlinearity**:
   Depth without nonlinearity is simply a product of matrices, collapsing into a single linear map. Nonlinearity bends the representational coordinate space, enabling non-convex decision boundaries.
2. **Evidence of useful learning signal vs. random non-zero gradient**:
   The loss consistently declined across training steps from $0.748 \to 0.0007$, and predictions cleanly polarized toward ground truth rather than drifting erratically.
3. **Why zero-initialization fails**:
   Symmetric weights cause identical forward responses and identical gradients, preventing the hidden neurons from specializing into distinct feature detectors.
4. **Activation impact: Science vs. Engineering**:
   - *Scientific*: Tanh is zero-centered and preserves gradient signs better; ReLU introduces non-saturating gradients for positive activations but risks complete inactivation in tiny networks.
   - *Engineering*: Choice of learning rate and numerical clipping determines whether dead neurons or exploding gradients destabilize training.
5. **Why match output layer and loss?**
   The loss must reflect the negative log-likelihood of the output distribution (e.g. Bernoulli for Sigmoid, Categorical for Softmax). Mismatched pairings (e.g. MSE with Sigmoid) cause severe vanishing gradients.
6. **LLM assistance vs. human verification**:
   The LLM generated the PyTorch scaffolding rapidly, but human verification was required to detect symmetry trapping and understand ReLU dead-zones on 4-point toy datasets.
7. **Scaling to large models**:
   - *Keep*: Batch loss tracking, parameter gradient norms, seed reproducibility, evaluation on validation metrics.
   - *Drop / Too expensive*: Printing complete weight gradient matrices, exhaustive parameter inspection across every single layer.
