# Bayesian Networks Lab: Autoregressive Language Models

This laboratory examines how probability theory and **Bayesian Networks** form the mathematical foundation of modern **Autoregressive Language Models**.

---

## 1. Mathematical Foundation

The joint probability of a sequence of tokens $X_1, X_2, \dots, X_T$ decomposes exactly via the probability chain rule:
$$P(X_1, X_2, \dots, X_T) = P(X_1) \prod_{t=2}^T P(X_t \mid X_1, \dots, X_{t-1})$$

An **autoregressive model** estimates these conditional probabilities and generates text token by token:
$$\text{Sample } X_t \sim P(X_t \mid X_1, \dots, X_{t-1})$$

---

## 2. Part I & II: Markov Assumptions & Bayesian Networks

### Question 1: Why is this decomposition useful for generating text?
The full joint distribution over sequences of arbitrary length is intractable to represent directly. The autoregressive chain-rule decomposition transforms the global sequence generation problem into a sequence of local, step-by-step next-token predictions conditioned on preceding context.

### Question 2: What independence assumption is made by the first-order network?
$$X_1 \longrightarrow X_2 \longrightarrow X_3 \longrightarrow \dots \longrightarrow X_T$$
The **first-order Markov assumption** states that the current word $X_t$ is conditionally independent of all earlier words given the immediately preceding word $X_{t-1}$:
$$P(X_t \mid X_1, \dots, X_{t-1}) = P(X_t \mid X_{t-1}) \iff X_t \perp (X_1, \dots, X_{t-2}) \mid X_{t-1}$$

---

## 3. Part III & IV: Dataset & Conditional Probability Tables (CPTs)

### Training Corpus
```text
the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug
```
With boundary markers: `"<START> ... <END>"`.

### Question 3: Conditional Probability Distributions

$$P(w_j \mid w_i) = \frac{C(w_i, w_j)}{\sum_k C(w_i, w_k)}$$

| Context ($w_i$) | Total Transitions | Next Token Probabilities $P(w_j \mid w_i)$ |
| :--- | :---: | :--- |
| **`the`** | 12 | `cat`: $3/12 = 0.25$, `dog`: $3/12 = 0.25$, `mat`: $2/12 \approx 0.17$, `rug`: $2/12 \approx 0.17$, `park`: $2/12 \approx 0.17$ |
| **`cat`** | 3 | `sat`: $2/3 \approx 0.67$, `ran`: $1/3 \approx 0.33$ |
| **`dog`** | 3 | `sat`: $2/3 \approx 0.67$, `ran`: $1/3 \approx 0.33$ |
| **`sat`** | 4 | `on`: $4/4 = 1.00$ |
| **`ran`** | 2 | `to`: $2/2 = 1.00$ |

#### Zero-Probability Transitions
Any pair not observed in the corpus has probability 0 (e.g., $P(\text{mat} \mid \text{cat}) = 0$, $P(\text{dog} \mid \text{sat}) = 0$, $P(\text{the} \mid \text{the}) = 0$).

---

## 4. Implementation & Verification

Code: [`bn_language_model.py`](file:///c:/Users/sarthak/Desktop/i/Acads-coding/AI_Labs/BNLab/bn_language_model.py).

### How to Run
```bash
python BNLab/bn_language_model.py
```

### Questions 4–7: Code Architecture Inspection
- **Q4: Where in the program are transition counts stored?**
  In `self.transition_counts`, a nested `defaultdict(Counter)`.
- **Q5: Where is $P(X_t \mid X_{t-1})$ computed?**
  During `train()`, where each count $C(w_{t-1}, w_t)$ is divided by the sum of row counts: `count / total`.
- **Q6: Greedy Selection vs. Sampling:**
  - *Greedy*: selects $\text{argmax}_w P(w \mid w_{\text{prev}})$. It is purely deterministic and mode-seeking.
  - *Sampling*: draws a random token weighted by $P(w \mid w_{\text{prev}})$ (`random.choices`). It explores the full distribution support.
- **Q7: What happens for an unobserved word?**
  The model returns an empty distribution `{}`. In `generate_sentence`, an unseen token safely defaults to `<END>` to prevent infinite loops.

### Part VII: Normalization Tests
$$\sum_{v \in V} P(v \mid w) = 1.0 \quad (\forall w)$$

```text
Context: <START>    | Sum = 1.000000 | Normalized: True
Context: the        | Sum = 1.000000 | Normalized: True
Context: cat        | Sum = 1.000000 | Normalized: True
Context: sat        | Sum = 1.000000 | Normalized: True
Context: on         | Sum = 1.000000 | Normalized: True
Context: mat        | Sum = 1.000000 | Normalized: True
Context: rug        | Sum = 1.000000 | Normalized: True
Context: dog        | Sum = 1.000000 | Normalized: True
Context: ran        | Sum = 1.000000 | Normalized: True
Context: to         | Sum = 1.000000 | Normalized: True
Context: park       | Sum = 1.000000 | Normalized: True
```

- **Q8: If one of the totals is 0.87, what does this tell you?**
  It indicates a defect in normalization: either missing outcomes from the vocabulary sum, floating-point truncation, or conditioning over incomplete events.

---

## 5. Part VIII & IX: Prediction and Generation

### Q9: Argmax Predictions vs. Human Expectations
- `the` $\to$ `cat` ($P=0.25$)
- `cat` $\to$ `sat` ($P=0.67$)
- `dog` $\to$ `sat` ($P=0.67$)
- `sat` $\to$ `on` ($P=1.00$)
- `ran` $\to$ `to` ($P=1.00$)

A statistical model simply reflects the empirical training frequencies. Humans expect grammatical syntax and semantics based on extensive world knowledge and global narrative context.

### Q10: Deterministic vs. Probabilistic Generation
- **Greedy Generation (5 runs)**:
  `the cat sat on the cat sat on the cat sat on the...` (Trapped in a deterministic 4-token cycle!)
- **Sampling Generation (5 runs)**:
  1. `the park`
  2. `the rug`
  3. `the mat`
  4. `the rug`
  5. `the cat sat on the cat sat on the park`

**Why?** The greedy policy always picks the highest single probability, getting stuck in an absorbing cycle when `the -> cat -> sat -> on -> the` recurs. Sampling explores the alternative paths (`dog`, `mat`, `rug`, `park`), producing diverse outputs.

---

## 6. Part XI–XIII: Second-Order Model vs. First-Order Model

The second-order model conditions on two previous words:
$$P(X_t \mid X_{t-2}, X_{t-1})$$
Bayesian Network structure:
$$X_{t-2} \longrightarrow X_t \longleftarrow X_{t-1}$$

### Comparison Table

| Metric | First-Order Model | Second-Order Model |
| :--- | :---: | :---: |
| **Conditioning Context** | 1 token ($X_{t-1}$) | 2 tokens ($X_{t-2}, X_{t-1}$) |
| **Observed Contexts** | 11 | 15 |
| **Non-Zero Parameters in CPT** | 17 | 19 |
| **Theoretical Context Space ($|V|^k$)** | $12^1 = 12$ | $12^2 = 144$ |
| **Zero-Probability Context Sparsity** | $8.3\%$ | **$89.6\%$** |
| **Qualitative Coherence** | Low (hallucinates cycles / nonsense) | **High (generates valid full sentences)** |

### Second-Order Generated Examples
1. `the cat ran to the park`
2. `the dog ran to the park`
3. `the dog ran to the park`
4. `the cat sat on the mat`
5. `the cat ran to the park`

### Q11 & Q12: Why More Context Improves Coherence but Increases Sparsity
- **Improved Prediction**: Conditioning on `(on, the)` allows the model to know it is describing a surface (`mat` or `rug`), preventing it from transitioning back to animals (`cat` or `dog`).
- **Data Sparsity**: The context space grows exponentially as $|V|^n$. In our small corpus, $89.6\%$ of 2-gram contexts were never observed, meaning an n-gram CPT rapidly runs out of statistical support as context length increases.

---

## 7. Part XIV & XV: Connection to Modern LLMs & Reflection

### Comparison: Hand-Built BN vs. Modern Transformer LLM

| Feature | Simple BN Model (n-gram) | Modern Autoregressive LLM (e.g. Gemini, GPT) |
| :--- | :--- | :--- |
| **Representation** | Explicit Conditional Probability Tables | Neural network weights (Self-Attention + MLP) |
| **Context** | Fixed window ($k=1$ or $k=2$) | Thousands of tokens learned dynamically |
| **Parameter Scaling** | Combinatorial table growth ($O(\|V\|^n)$) | Fixed weight matrices parameterized across layers |
| **Generalization** | Exact matching; zero probability for unseen pairs | Continuous dense embeddings; smooth generalization |
| **Learning** | Counting and normalizing frequencies | Gradient descent backpropagation on cross-entropy |

### Question 13: Prompting Approach A vs. Approach B
**Approach B** is vastly superior. Specifying the exact mathematical model $P(X_t \mid X_{t-1})$, counting mechanisms, and validation criteria ensures that the LLM implements the intended AI science rather than generating arbitrary boilerplate with external libraries.

### Question 14: What did thinking of language models as Bayesian Networks provide?
1. **Factorization**: Proves how autoregressive generation is mathematically derived from the probability chain rule.
2. **Independence Assumptions**: Clarifies why n-gram models lose long-range discourse structure.
3. **Falsifiable Testing**: Enables explicit verification of probability axioms ($\sum P = 1.0$).
