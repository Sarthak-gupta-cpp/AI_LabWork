"""
Artificial Intelligence - Neural Models Laboratory
Neural Models: Learning, Depth, Activations, and Output Layers

This module implements:
1. Linear Baseline (Single affine layer) demonstrating linear inseparability of XOR.
2. 2-2-1 Neural Model with PyTorch (Binary Classification via BCEWithLogitsLoss).
3. Backpropagation and Gradient Inspection (dL / dW^(1)).
4. Symmetry Breaking Experiment (Zero Weight Initialization vs Random Initialization).
5. Activation Function Comparative Benchmark (Sigmoid vs Tanh vs ReLU).
6. Three-Class Multiclass Extension with Softmax and Cross-Entropy Loss.
7. Numerical Stability & Shift Invariance Diagnostic for Softmax.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple


# Task 1 & 2 Dataset: XOR Safety Sensor Problem
X_XOR = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)

Y_XOR = torch.tensor([[0.0],
                      [1.0],
                      [1.0],
                      [0.0]], dtype=torch.float32)

# Task 5 Dataset: Three-Class Safety Sensor Decision
# Class 0: (0,0) - both inactive
# Class 1: (0,1) or (1,0) - disagreement
# Class 2: (1,1) - both active
Y_3CLASS = torch.tensor([0, 1, 1, 2], dtype=torch.long)


class LinearBaseline(nn.Module):
    """Single affine layer without hidden non-linearity."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)


class XORNet(nn.Module):
    """2-2-1 Neural Network for XOR decision boundary."""
    def __init__(self, activation: str = "sigmoid"):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out


class ThreeClassNet(nn.Module):
    """2 -> 4 -> 3 Neural Network for three-class sensor status."""
    def __init__(self, hidden_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        h = self.act(self.fc1(x))
        logits = self.fc2(h)
        return logits


def train_linear_model() -> Tuple[float, torch.Tensor]:
    """Demonstrates that a linear model fails to solve XOR."""
    torch.manual_seed(42)
    model = LinearBaseline()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for _ in range(2000):
        optimizer.zero_grad()
        logits = model(X_XOR)
        loss = criterion(logits, Y_XOR)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = torch.sigmoid(model(X_XOR))
    return loss.item(), preds


def run_xor_experiment() -> Dict:
    """Task 4 Part A & B: Standard training and gradient inspection."""
    torch.manual_seed(42)
    model = XORNet(activation="sigmoid")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # Initial loss
    with torch.no_grad():
        init_loss = criterion(model(X_XOR), Y_XOR).item()

    early_grad_norm = None
    for step in range(1, 2001):
        optimizer.zero_grad()
        logits = model(X_XOR)
        loss = criterion(logits, Y_XOR)
        loss.backward()

        if step == 10:
            early_grad_norm = model.fc1.weight.grad.norm(2).item()

        optimizer.step()

    final_loss = loss.item()
    with torch.no_grad():
        final_logits = model(X_XOR)
        final_probs = torch.sigmoid(final_logits)
        final_preds = (final_probs >= 0.5).float()

    return {
        "model": model,
        "init_loss": init_loss,
        "final_loss": final_loss,
        "final_probs": final_probs,
        "final_preds": final_preds,
        "fc1_grad": model.fc1.weight.grad.clone(),
        "early_grad_norm": early_grad_norm
    }


def run_symmetry_experiment() -> Dict:
    """Task 4 Part C: Zero weight initialization failure."""
    torch.manual_seed(42)
    model = XORNet(activation="sigmoid")

    # Initialize all weights and biases to exactly zero
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    row_history = []
    for step in range(1, 201):
        optimizer.zero_grad()
        logits = model(X_XOR)
        loss = criterion(logits, Y_XOR)
        loss.backward()
        optimizer.step()

        if step in [1, 5, 20, 100, 200]:
            w1 = model.fc1.weight.detach().clone()
            row_history.append((step, w1[0].tolist(), w1[1].tolist()))

    with torch.no_grad():
        final_probs = torch.sigmoid(model(X_XOR))

    return {
        "row_history": row_history,
        "final_probs": final_probs,
        "final_loss": loss.item()
    }


def run_activation_benchmark() -> Dict[str, Dict]:
    """Task 4 Part D: Comparing Sigmoid, Tanh, and ReLU."""
    activations = ["sigmoid", "tanh", "relu"]
    results = {}

    for act in activations:
        torch.manual_seed(42)
        model = XORNet(activation=act)
        criterion = nn.BCEWithLogitsLoss()
        # Use SGD with momentum to evaluate standard first-order optimization dynamics
        optimizer = optim.SGD(model.parameters(), lr=0.15, momentum=0.9)

        early_grad_norm = 0.0
        for step in range(1, 3001):
            optimizer.zero_grad()
            logits = model(X_XOR)
            loss = criterion(logits, Y_XOR)
            loss.backward()

            if step == 10:
                early_grad_norm = model.fc1.weight.grad.norm(2).item()

            optimizer.step()

        with torch.no_grad():
            final_probs = torch.sigmoid(model(X_XOR))
            preds = (final_probs >= 0.5).float()
            all_correct = bool(torch.equal(preds, Y_XOR))

        results[act] = {
            "final_loss": loss.item(),
            "all_correct": all_correct,
            "early_grad_norm": early_grad_norm,
            "probs": final_probs.squeeze().tolist()
        }

    return results


def run_three_class_experiment() -> Dict:
    """Task 5: Multiclass extension with CrossEntropyLoss and Shift Invariance test."""
    torch.manual_seed(42)
    model = ThreeClassNet(hidden_dim=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    for _ in range(2000):
        optimizer.zero_grad()
        logits = model(X_XOR)
        loss = criterion(logits, Y_3CLASS)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X_XOR)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        # Shift invariance diagnostic: Add constant c = 100 to all logits
        shifted_logits = logits + 100.0
        shifted_probs = torch.softmax(shifted_logits, dim=-1)
        max_diff = torch.max(torch.abs(probs - shifted_probs)).item()

    return {
        "final_loss": loss.item(),
        "logits": logits,
        "probs": probs,
        "preds": preds,
        "max_diff": max_diff,
        "fc2_weight_shape": list(model.fc2.weight.shape)
    }


def execute_all_neural_lab_tasks():
    print("=" * 75)
    print("AI LAB - NEURAL MODELS: LEARNING, DEPTH, ACTIVATIONS & OUTPUT LAYERS")
    print("=" * 75)

    # 1. Linear Baseline
    print("\n[TASK 1: LINEAR BASELINE INSEPARABILITY TEST]")
    lin_loss, lin_preds = train_linear_model()
    print(f"Single Affine + Sigmoid Final Loss: {lin_loss:.4f}")
    print(f"Predictions:\n{lin_preds.squeeze().tolist()}")
    print("Conclusion: Linear model fails on XOR (predicts ~0.5 for all inputs, max accuracy 50-75%).")

    # 2. 2-2-1 XOR Model (Tasks 3 & 4A, 4B)
    print("\n" + "=" * 75)
    print("[TASK 4 PART A & B: XOR 2-2-1 TRAINING & BACKPROPAGATION CHECK]")
    res_xor = run_xor_experiment()
    print(f"Initial Loss : {res_xor['init_loss']:.4f}")
    print(f"Final Loss   : {res_xor['final_loss']:.6f}")
    print("\nTarget vs Predicted Probabilities:")
    for i in range(4):
        x = X_XOR[i].tolist()
        y = int(Y_XOR[i].item())
        prob = res_xor['final_probs'][i].item()
        pred = int(res_xor['final_preds'][i].item())
        print(f"  Input {x} -> Target: {y} | Model Prob: {prob:.4f} | Prediction: {pred}")

    print("\nFirst-Layer Weight Gradient Tensor (dL / dW^(1)):")
    print(res_xor['fc1_grad'])
    print("Interpretation: parameter.grad holds accumulated dL/dW^(1) averaged over all 4 batch samples.")

    # 3. Symmetry Experiment (Task 4 Part C)
    print("\n" + "=" * 75)
    print("[TASK 4 PART C: ZERO-WEIGHT INITIALIZATION & SYMMETRY FAILURE]")
    res_sym = run_symmetry_experiment()
    print("Hidden Unit Weights across Training Steps:")
    for step, r1, r2 in res_sym['row_history']:
        print(f"  Step {step:3d} | Unit 1: {r1} | Unit 2: {r2} | Identical: {r1 == r2}")
    print(f"Final Loss under Zero Init: {res_sym['final_loss']:.4f}")
    print("Explanation: Zero weights cause hidden units to compute identical activations and gradients.")
    print("             Symmetry is NEVER broken; the network collapses into a single unit.")

    # 4. Activation Benchmark (Task 4 Part D)
    print("\n" + "=" * 75)
    print("[TASK 4 PART D: ACTIVATION EXPERIMENT BENCHMARK]")
    res_act = run_activation_benchmark()
    print(f"{'Hidden Activation':<18} | {'Final Loss':<12} | {'4/4 Correct?':<14} | {'Early ||grad W^(1)||_2':<22}")
    print("-" * 75)
    for act_name, data in res_act.items():
        print(f"{act_name.capitalize():<18} | {data['final_loss']:<12.6f} | {str(data['all_correct']):<14} | {data['early_grad_norm']:<22.6f}")

    # 5. Three-Class Extension (Task 5)
    print("\n" + "=" * 75)
    print("[TASK 5: THREE-CLASS EXTENSION & SOFTMAX NUMERICAL DIAGNOSTIC]")
    res_3class = run_three_class_experiment()
    print(f"Final Multiclass Cross-Entropy Loss: {res_3class['final_loss']:.6f}")
    print(f"Final Weight Matrix Shape: {res_3class['fc2_weight_shape']} (3 logits x 4 hidden units)")
    print("\nThree-Class Predictions:")
    for i in range(4):
        x = X_XOR[i].tolist()
        y = int(Y_3CLASS[i].item())
        probs = res_3class['probs'][i].tolist()
        pred = int(res_3class['preds'][i].item())
        prob_sum = sum(probs)
        print(f"  Input {x} -> True Class: {y} | Pred: {pred} | Probabilities: [{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}] (Sum = {prob_sum:.6f})")

    print("\nSoftmax Shift Invariance Diagnostic (Adding constant c = +100 to all logits):")
    print(f"Maximum absolute difference between original and shifted softmax: {res_3class['max_diff']:.2e}")
    print("Conclusion: Softmax is mathematically invariant to constant logit shifts.")
    print("=" * 75)


if __name__ == "__main__":
    execute_all_neural_lab_tasks()
