# Logic Lab: Logical Reasoning for Planning

This laboratory investigates the integration of formal logic and state-space search to construct a STRIPS-style planning agent:
$$\text{Logic} + \text{Search} = \text{Planning}$$

---

## 1. Learning Objectives

- Formulate automated planning problems using propositional states $(I, A, G)$, actions, preconditions, and effects.
- Determine action applicability using logical entailment: $S \models \text{Preconditions}(a)$.
- Apply state transition updates: $S' = (S \setminus \text{NegEffects}(a)) \cup \text{PosEffects}(a)$.
- Implement a Breadth-First Search (BFS) forward state-space planner in Python.
- Perform independent formal verification of generated plans using Python and Prolog.
- Analyze the boundary between statistical code generation (LLMs) and rigorous logical verification.

---

## 2. Motivating Scenario

A small warehouse robot transports packages across three locations arranged linearly:
$$A \longleftrightarrow B \longleftrightarrow C$$

- **Initial state**: $I = \{ \text{At(Robot, A)}, \text{At(Package, A)} \}$
- **Goal state**: $G = \{ \text{At(Package, C)} \}$

---

## 3. Task 0: Understanding the Planning Problem

### (a) Initial State $I$
$$I = \{ \text{At(Robot, A)}, \text{At(Package, A)} \}$$

### (b) Goal State $G$
$$G = \{ \text{At(Package, C)} \}$$

### (c) & (d) Action Schema

| Action | Positive Preconditions | Negative Preconditions | Positive Effects | Negative Effects |
| :--- | :--- | :--- | :--- | :--- |
| $\text{Move}(X, Y)$ | $\text{At(Robot, } X\text{)}$ | $\emptyset$ | $\text{At(Robot, } Y\text{)}$ | $\text{At(Robot, } X\text{)}$ |
| $\text{PickUp}(\text{Package}, X)$ | $\text{At(Robot, } X\text{)}, \text{At(Package, } X\text{)}$ | $\emptyset$ | $\text{Holding(Package)}$ | $\text{At(Package, } X\text{)}$ |
| $\text{Drop}(\text{Package}, X)$ | $\text{At(Robot, } X\text{)}, \text{Holding(Package)}$ | $\emptyset$ | $\text{At(Package, } X\text{)}$ | $\text{Holding(Package)}$ |

### Applicability Check from $I$:
- **Is $\text{PickUp}(\text{Package}, A)$ applicable?**
  **Yes.** Its preconditions are $\{\text{At(Robot, A)}, \text{At(Package, A)}\}$, both of which belong to $I$.
- **Is $\text{Drop}(\text{Package}, C)$ applicable?**
  **No.** Its preconditions are $\{\text{At(Robot, C)}, \text{Holding(Package)}\}$, neither of which is present in $I$.

---

## 4. Task 1: Manual Plan Construction

A valid plan is a grounded action sequence $\langle a_1, a_2, \dots, a_n \rangle$ such that $I \xrightarrow{a_1} S_1 \xrightarrow{a_2} \dots \xrightarrow{a_n} S_n$ where $S_n \models G$:

| Step | Action | State After Action ($S_k$) |
| :---: | :--- | :--- |
| **$S_0$** | *(Initial)* | $\{\text{At(Robot, A)}, \text{At(Package, A)}\}$ |
| **$S_1$** | $\text{PickUp}(\text{Package}, A)$ | $\{\text{At(Robot, A)}, \text{Holding(Package)}\}$ |
| **$S_2$** | $\text{Move}(A, B)$ | $\{\text{At(Robot, B)}, \text{Holding(Package)}\}$ |
| **$S_3$** | $\text{Move}(B, C)$ | $\{\text{At(Robot, C)}, \text{Holding(Package)}\}$ |
| **$S_4$** | $\text{Drop}(\text{Package}, C)$ | $\{\text{At(Robot, C)}, \text{At(Package, C)}\}$ |

At $S_4$, $\text{At(Package, C)} \in S_4 \implies S_4 \models G$. The plan is sound.

---

## 5. Task 2 & 3: Python Implementation & Testing

The code is implemented in [`logic_planner.py`](file:///c:/Users/sarthak/Desktop/i/Acads-coding/AI_Labs/LogicLab/logic_planner.py).

### How to Run
```bash
python LogicLab/logic_planner.py
```

### Execution Results

```text
======================================================================
ARTIFICIAL INTELLIGENCE - LOGIC PLANNING LAB
======================================================================

[TEST A: SOLVABLE PROBLEM]
Initial State : ['At(Package, A)', 'At(Robot, A)']
Goal State    : ['At(Package, C)']
Plan Found?   : True
Actions in Plan (4):
  Step 1: PickUp(Package, A)
  Step 2: Move(A, B)
  Step 3: Move(B, C)
  Step 4: Drop(Package, C)

State Progression:
  S0: ['At(Package, A)', 'At(Robot, A)']
  S1: ['At(Robot, A)', 'Holding(Package)']
  S2: ['At(Robot, B)', 'Holding(Package)']
  S3: ['At(Robot, C)', 'Holding(Package)']
  S4: ['At(Package, C)', 'At(Robot, C)']
Independent Plan Validation: VALID (Plan independently verified: all preconditions and goals satisfied.)

----------------------------------------------------------------------
[TEST B: IMPOSSIBLE PROBLEM (PickUp Action Removed)]
Initial State : ['At(Package, A)', 'At(Robot, A)']
Goal State    : ['At(Package, C)']
Plan Found?   : False
Planner Output: No plan found

----------------------------------------------------------------------
[TEST C: IRRELEVANT / DISTRACTOR ACTIONS]
Initial State : ['At(Package, A)', 'At(Robot, B)']
Goal State    : ['At(Package, C)']
Plan Found?   : True
Plan Steps (5):
  Step 1: Move(B, A)
  Step 2: PickUp(Package, A)
  Step 3: Move(A, B)
  Step 4: Move(B, C)
  Step 5: Drop(Package, C)
Independent Plan Validation: VALID
======================================================================
```

### Test Summary Table

| Test Case | Initial State | Goal | Plan Found? | Actions Returned | Validated? |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **Test A: Solvable** | $\{\text{At(R, A)}, \text{At(P, A)}\}$ | $\text{At(P, C)}$ | `True` | 4 steps (PickUp $\to$ Move $\to$ Move $\to$ Drop) | Yes |
| **Test B: Impossible** | $\{\text{At(R, A)}, \text{At(P, A)}\}$ | $\text{At(P, C)}$ | `False` | None (`No plan found`) | Yes |
| **Test C: Distractor** | $\{\text{At(R, B)}, \text{At(P, A)}\}$ | $\text{At(P, C)}$ | `True` | 5 steps (Move to A $\to$ PickUp $\to$ Move $\to$ Move $\to$ Drop) | Yes |

---

## 6. Task 4: Logic and Search

### Completed Planning Loop
```text
Current state S
      ↓
Check action preconditions (Logic: S |= Preconditions(a))
      ↓
Applicable actions
      ↓
Generate successor state (Transition: S' = Apply(S, a))
      ↓
Search over alternatives (BFS queue / frontier)
      ↓
Goal satisfied? (Logic: S' |= G)
```

- **Logic determines what is possible**: It evaluates state constraints, checks preconditions, and executes consistent belief updates.
- **Search determines what to try**: It navigates the tree of applicable actions to find a path to the goal.

---

## 7. Task 5: Verifying Generated Plans

### Which should you trust more?
**(b) The independently executed state transitions.**
- **Reason**: An LLM is a probabilistic token predictor. It can output plausible-sounding text claiming that preconditions hold even when they conflict with the environment's physics (hallucination). A formal verification engine computes explicit set membership ($S \models \text{Preconditions}(a)$) and is mathematically deterministic and falsifiable.

---

## 8. Section 7: Optional Extension — Prolog as a Logical Verifier

Files:
- [`planner.pl`](file:///c:/Users/sarthak/Desktop/i/Acads-coding/AI_Labs/LogicLab/planner.pl)
- [`prolog_verifier.py`](file:///c:/Users/sarthak/Desktop/i/Acads-coding/AI_Labs/LogicLab/prolog_verifier.py)

### Running the Verifier
```bash
python LogicLab/prolog_verifier.py
```

### Output
```text
============================================================
PROLOG LOGICAL VERIFICATION TEST SUITE (Section 7)
============================================================

[Task 6 Queries]
?- can_move(a, b).  -->  True (Direct edge exists)
?- can_move(a, c).  -->  False (No direct edge exists)

[Task 7 Queries: Verifying Proposed Plan Moves]
?- valid_move(a, b).  -->  True (SUCCEEDED)
?- valid_move(b, c).  -->  True (SUCCEEDED)
?- valid_move(a, c).  -->  False (FAILED)

[Task 8: Logical Reasoning Chain]
Fact  : wet_road.
Rule 1: slippery :- wet_road.
Rule 2: reduce_speed :- slippery.
Query : ?- reduce_speed.  -->  True
Proof : wet_road => slippery => reduce_speed
============================================================
```

### Prolog Questions & Answers
- **(a) Why does Prolog return `true` for `can_move(a, b)`?**
  Because the fact `connected(a, b)` exists in the database, satisfying the body of `can_move(X, Y) :- connected(X, Y)`.
- **(b) Why does it not establish `can_move(a, c)`?**
  Because there is no fact `connected(a, c)` in the database (Closed World Assumption).
- **(c) Logical implication**:
  $$\text{Connected}(X, Y) \longrightarrow \text{CanMove}(X, Y)$$

---

## 9. Reflection Questions (Section 5 & 7.2)

1. **Why specify action preconditions and effects before asking an LLM to code?**
   Formalizing the STRIPS schema grounds the prompt with exact mathematical invariants, eliminating ambiguity about state representation and transitions.
2. **Example error if preconditions are ignored**:
   The planner might output `Drop(Package, C)` when the robot is still at $A$ and never picked up the package.
3. **Why a "reasonable-looking" plan may not be valid**:
   Natural language heuristics frequently skip hidden prerequisites (e.g. holding the package while moving, or moving between unconnected nodes).
4. **LLM contribution vs. human verification**:
   The LLM efficiently scaffolds the data structures and BFS loop, but human/formal verification ensures termination guarantees and state consistency.
5. **Difference between a Prolog fact and rule**:
   A *fact* asserts an unconditional proposition (e.g., `connected(a, b).`), while a *rule* asserts a conditional implication containing a head and body (e.g., `can_move(X, Y) :- connected(X, Y).`).
