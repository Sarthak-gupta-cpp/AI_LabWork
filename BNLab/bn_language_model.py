"""
Artificial Intelligence - Bayesian Networks Laboratory
Bayesian Networks and Autoregressive Language Models

This module implements:
1. Tokenization and Corpus Preprocessing with <START> and <END> tokens.
2. First-Order Autoregressive Language Model (First-Order Markov Bayesian Network).
3. Second-Order Autoregressive Language Model (Second-Order Markov Bayesian Network).
4. Probability Normalization Testing: sum_v P(v | context) == 1.0.
5. Deterministic (Greedy / Argmax) vs. Probabilistic (Sampling) Text Generation.
6. Empirical comparison: parameter counts, zero-probability contexts, and sample outputs.
"""

import random
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


DATASET = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug"
]


class FirstOrderLanguageModel:
    """
    First-Order Autoregressive Language Model.
    Represents the Bayesian Network: X_1 -> X_2 -> X_3 -> ... -> X_T
    Models: P(X_t | X_{t-1}) = C(w_{t-1}, w_t) / sum_k C(w_{t-1}, w_k)
    """
    def __init__(self):
        # transition_counts[w_prev][w_curr] = count
        self.transition_counts: Dict[str, Counter] = defaultdict(Counter)
        # conditional_probs[w_prev][w_curr] = probability
        self.conditional_probs: Dict[str, Dict[str, float]] = {}
        self.vocabulary = set()

    def train(self, sentences: List[str]):
        self.transition_counts.clear()
        self.conditional_probs.clear()
        self.vocabulary.clear()

        for sentence in sentences:
            tokens = ["<START>"] + sentence.lower().strip().split() + ["<END>"]
            self.vocabulary.update(tokens)
            for i in range(len(tokens) - 1):
                w_prev = tokens[i]
                w_curr = tokens[i + 1]
                self.transition_counts[w_prev][w_curr] += 1

        # Compute Conditional Probability Table (CPT)
        for w_prev, next_counts in self.transition_counts.items():
            total = sum(next_counts.values())
            self.conditional_probs[w_prev] = {
                w_curr: count / total for w_curr, count in next_counts.items()
            }

    def get_distribution(self, prev_word: str) -> Dict[str, float]:
        return self.conditional_probs.get(prev_word, {})

    def predict_most_probable(self, prev_word: str) -> Optional[Tuple[str, float]]:
        dist = self.get_distribution(prev_word)
        if not dist:
            return None
        best_word = max(dist, key=dist.get)
        return best_word, dist[best_word]

    def verify_normalization(self, tolerance: float = 1e-5) -> List[Tuple[str, float, bool]]:
        results = []
        for word, dist in self.conditional_probs.items():
            total = sum(dist.values())
            is_valid = abs(total - 1.0) < tolerance
            results.append((word, total, is_valid))
        return results

    def sample_next_token(self, prev_word: str) -> str:
        dist = self.get_distribution(prev_word)
        if not dist:
            return "<END>"
        words = list(dist.keys())
        probs = list(dist.values())
        return random.choices(words, weights=probs, k=1)[0]

    def generate_sentence(self, mode: str = "sampling", max_tokens: int = 25) -> str:
        tokens = []
        current = "<START>"

        for _ in range(max_tokens):
            if mode == "greedy":
                pred = self.predict_most_probable(current)
                if not pred:
                    break
                next_token = pred[0]
            else:
                next_token = self.sample_next_token(current)

            if next_token == "<END>":
                break
            tokens.append(next_token)
            current = next_token

        return " ".join(tokens)


class SecondOrderLanguageModel:
    """
    Second-Order Autoregressive Language Model.
    Represents the Bayesian Network: (X_{t-2}, X_{t-1}) -> X_t
    Models: P(X_t | X_{t-2}, X_{t-1})
    """
    def __init__(self):
        # counts[(w_{t-2}, w_{t-1})][w_t] = count
        self.transition_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.conditional_probs: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.vocabulary = set()

    def train(self, sentences: List[str]):
        self.transition_counts.clear()
        self.conditional_probs.clear()
        self.vocabulary.clear()

        for sentence in sentences:
            tokens = ["<START>", "<START>"] + sentence.lower().strip().split() + ["<END>"]
            self.vocabulary.update(tokens)
            for i in range(len(tokens) - 2):
                context = (tokens[i], tokens[i + 1])
                w_curr = tokens[i + 2]
                self.transition_counts[context][w_curr] += 1

        for context, next_counts in self.transition_counts.items():
            total = sum(next_counts.values())
            self.conditional_probs[context] = {
                w_curr: count / total for w_curr, count in next_counts.items()
            }

    def get_distribution(self, context: Tuple[str, str]) -> Dict[str, float]:
        return self.conditional_probs.get(context, {})

    def sample_next_token(self, context: Tuple[str, str]) -> str:
        dist = self.get_distribution(context)
        if not dist:
            return "<END>"
        words = list(dist.keys())
        probs = list(dist.values())
        return random.choices(words, weights=probs, k=1)[0]

    def generate_sentence(self, mode: str = "sampling", max_tokens: int = 25) -> str:
        tokens = []
        w1, w2 = "<START>", "<START>"

        for _ in range(max_tokens):
            dist = self.get_distribution((w1, w2))
            if not dist:
                break

            if mode == "greedy":
                next_token = max(dist, key=dist.get)
            else:
                next_token = self.sample_next_token((w1, w2))

            if next_token == "<END>":
                break
            tokens.append(next_token)
            w1, w2 = w2, next_token

        return " ".join(tokens)


def run_laboratory_experiments():
    random.seed(42)
    print("=" * 75)
    print("AI LAB - BAYESIAN NETWORKS & AUTOREGRESSIVE LANGUAGE MODELS")
    print("=" * 75)

    # 1. First-Order Model Training
    lm1 = FirstOrderLanguageModel()
    lm1.train(DATASET)

    print(f"\n[PART IV: CONDITIONAL PROBABILITY TABLES (First-Order Model)]")
    focus_words = ["the", "cat", "dog", "sat", "ran", "<START>"]
    for w in focus_words:
        dist = lm1.get_distribution(w)
        dist_str = ", ".join([f"P({k}|{w})={v:.2f}" for k, v in dist.items()])
        print(f"  Context '{w}': {dist_str}")

    # 2. Probability Normalization Testing (Part VII)
    print("\n[PART VII: PROBABILITY NORMALIZATION TESTS (sum_v P(v | w) == 1.0)]")
    norm_results = lm1.verify_normalization()
    for word, total, valid in norm_results:
        print(f"  Context: {word:<10} | Sum = {total:.6f} | Normalized: {valid}")

    # 3. Next-Word Prediction (Part VIII)
    print("\n[PART VIII: PREDICTING THE MOST PROBABLE NEXT WORD (ARGMAX)]")
    test_contexts = ["the", "cat", "dog", "sat", "ran"]
    for ctx in test_contexts:
        pred = lm1.predict_most_probable(ctx)
        if pred:
            word, prob = pred
            print(f"  Context: '{ctx}' -> argmax: '{word}' (P = {prob:.2f})")

    # 4. Text Generation (Part IX: 20 Sentences via Sampling)
    print("\n[PART IX: GENERATING 20 SENTENCES VIA SAMPLING (First-Order)]")
    generated_20 = [lm1.generate_sentence(mode="sampling") for _ in range(20)]
    for idx, s in enumerate(generated_20, 1):
        print(f"  {idx:2d}. {s}")

    # 5. Deterministic (Greedy) vs Probabilistic (Sampling) (Part X)
    print("\n[PART X: DETERMINISTIC (GREEDY) VS PROBABILISTIC (SAMPLING)]")
    print("--- Mode A: Greedy Generation (5 runs) ---")
    for i in range(5):
        print(f"  Greedy {i+1}: {lm1.generate_sentence(mode='greedy')}")

    print("--- Mode B: Sampling Generation (5 runs) ---")
    for i in range(5):
        print(f"  Sampling {i+1}: {lm1.generate_sentence(mode='sampling')}")

    # 6. Second-Order Model Training & Comparison (Part XI - XIII)
    print("\n[PART XI - XIII: SECOND-ORDER MODEL & COMPARISON]")
    lm2 = SecondOrderLanguageModel()
    lm2.train(DATASET)

    # Distinct parameters: total entries in CPT
    params_lm1 = sum(len(d) for d in lm1.conditional_probs.values())
    params_lm2 = sum(len(d) for d in lm2.conditional_probs.values())

    # Total possible contexts
    vocab_size = len(lm1.vocabulary)
    possible_contexts_lm1 = vocab_size
    possible_contexts_lm2 = vocab_size ** 2

    observed_contexts_lm1 = len(lm1.conditional_probs)
    observed_contexts_lm2 = len(lm2.conditional_probs)
    zero_prob_contexts_lm2 = possible_contexts_lm2 - observed_contexts_lm2

    print(f"Vocabulary Size (|V|): {vocab_size}")
    print(f"First-Order Model : {params_lm1} non-zero CPT parameters across {observed_contexts_lm1} observed contexts.")
    print(f"Second-Order Model: {params_lm2} non-zero CPT parameters across {observed_contexts_lm2} observed contexts.")
    print(f"Second-Order Zero-Probability Contexts: {zero_prob_contexts_lm2} / {possible_contexts_lm2} ({(zero_prob_contexts_lm2/possible_contexts_lm2)*100:.1f}% sparse)")

    print("\n--- Second-Order Generated Sentences (5 samples) ---")
    for i in range(5):
        print(f"  Sample {i+1}: {lm2.generate_sentence(mode='sampling')}")

    # Save outputs to file
    with open("BNLab/generated_sentences.txt", "w") as f:
        f.write("=== 20 FIRST-ORDER SAMPLED SENTENCES ===\n")
        for idx, s in enumerate(generated_20, 1):
            f.write(f"{idx}. {s}\n")
        f.write("\n=== 5 SECOND-ORDER SAMPLED SENTENCES ===\n")
        for i in range(5):
            f.write(f"{i+1}. {lm2.generate_sentence(mode='sampling')}\n")

    print("\nResults and generated sentences saved to BNLab/generated_sentences.txt")
    print("=" * 75)


if __name__ == "__main__":
    run_laboratory_experiments()
