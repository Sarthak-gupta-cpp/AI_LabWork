# Search Lab: Search and A*

This laboratory focuses on formulating, planning, implementing, testing, and evaluating informed and uninformed search algorithms ($A^*$ and BFS) for autonomous warehouse navigation.

---

## 1. Problem Formulation: Warehouse Robot Navigation

The robot operates on a grid with obstacles (`#`), traversable cells (`.`), start position `S`, and goal position `G`.

```text
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
```

### Formal Problem Specification: $P = (S, A, T, s_0, G, c)$

| Component | Specification |
| :--- | :--- |
| **State Space $S$** | Discrete 2D grid coordinates $(r, c)$ where $0 \le r < 9$, $0 \le c < 17$, and $\text{grid}[r][c] \ne \text{'\#'}$. |
| **Action Space $A$** | $\left\{ \text{Up}, \text{Down}, \text{Left}, \text{Right} \right\}$ |
| **Transition Function $T(s, a)$** | Deterministic coordinate translation: <br>• $\text{Up}: (r-1, c)$<br>• $\text{Down}: (r+1, c)$<br>• $\text{Left}: (r, c-1)$<br>• $\text{Right}: (r, c+1)$ |
| **Initial State $s_0$** | $(1, 1)$ corresponding to `S` |
| **Goal States $G$** | $\left\{ (7, 15) \right\}$ corresponding to `G` |
| **Path Cost Function $c(s, a, s')$** | Uniform step cost: $c(s, a, s') = 1$ for all valid moves. |

### Task 0 Questions
- **(a) What information is necessary to specify a state?**
  Only the agent's spatial coordinates $(r, c)$ on the grid.
- **(b) What makes an action invalid?**
  An action is invalid if the target cell falls outside the warehouse boundary or lands on an obstacle wall (`#`).
- **(c) Is this a deterministic search problem?**
  Yes. Given any state $s$ and action $a$, the successor state $s' = T(s, a)$ is uniquely determined with probability 1.
- **(d) What would constitute a solution?**
  A finite sequence of valid actions $\langle a_1, a_2, \dots, a_k \rangle$ that transitions the agent from $s_0$ to a state in $G$.

---

## 2. Agent Design

1. **State Representation**: Python integer tuple `(r, c)`.
2. **Warehouse Representation**: 2D list of characters `List[List[str]]` storing cell types.
3. **Valid Actions**: Filter cardinal moves against grid bounds and non-wall cells (`grid[r][c] != '#'`).
4. **Goal Recognition**: Direct coordinate equality check: `current_state == goal_state`.
5. **Frontier Structure**: Min-priority heap (`heapq`) storing tuples: `(f_score, g_score, counter, current_state, path, actions)` where `counter` breaks priority ties deterministically.
6. **Path Reconstruction**: Maintained incrementally as an action/state history along the search tree branch.

---

## 3. Implementation

The implementation is located in [`search_agent.py`](file:///c:/Users/sarthak/Desktop/i/Acads-coding/AI_Labs/SearchLab/search_agent.py).

### How to Run
```bash
python SearchLab/search_agent.py
```

---

## 4. Task 3: Systematic Test Results

Four test configurations were executed:

### Test 1: Original Warehouse Map (Serpentine Labyrinth)
- **Solution Found**: `True`
- **Optimal Path Length**: `40` steps
- **States Expanded**: `64`
- **Actions**: `Right -> Right -> Right -> Right -> Down -> Down -> Down -> Down -> Right -> Right -> Right -> Right -> Right -> Right -> Right -> Right -> Up -> Up -> Left -> Left -> Left -> Left -> Left -> Left -> Up -> Up -> Right -> Right -> Right -> Right -> Right -> Right -> Right -> Right -> Down -> Down -> Down -> Down -> Down -> Down`
- **Visualized Trajectory**:
```text
#################
#S****#*********#
#.###*#*#######*#
#...#*#*******#*#
###.#*#######*#*#
#...#*********#*#
#.###########.#*#
#.............#G#
#################
```

### Test 2: Trivial Case (Immediate Goal)
- Map:
  ```text
  #####
  #SG##
  #####
  ```
- **Solution Found**: `True`
- **Path Length**: `1` (`['Right']`)
- **States Expanded**: `2`

### Test 3: No Solution (Inaccessible Goal)
- Map:
  ```text
  #######
  #S....#
  ###.###
  #...#G#
  #######
  ```
- **Solution Found**: `False`
- **States Expanded**: `9`
- **Verification**: Frontier exhausted cleanly without infinite looping.

### Test 4: Alternative Paths
- Map:
  ```text
  #######
  #S...G#
  #.###.#
  #.....#
  #######
  ```
- **Solution Found**: `True`
- **Path Length**: `4` steps (Shortest direct top route chosen: `Right -> Right -> Right -> Right`).

---

## 5. Task 4: Code Concept Inspection

| Concept | Code Location in `search_agent.py` |
| :--- | :--- |
| **State** | Coordinate tuple `(r, c)` in `current_state` |
| **Action** | Action strings (`"Up"`, `"Down"`, `"Left"`, `"Right"`) |
| **Transition** | Coordinate arithmetic in `get_successors()` |
| **Goal test** | `if current == goal:` checked upon node expansion |
| **$g(n)$** | `g` and `g_scores` tracking exact cost from start |
| **$h(n)$** | `heuristic_fn(pos, goal)` |
| **$f(n)$** | `f_score = new_g + h` |
| **Frontier** | Priority heap `frontier` via `heapq.heappush` and `heapq.heappop` |
| **Visited states** | Set `visited` containing closed states |
| **Path reconstruction** | `path + [next_state]` and `actions + [action]` |

### Questions
- **(a) What data structure is used for the $A^*$ frontier?**
  A binary min-heap (`heapq` module in Python).
- **(b) How does the program select the next state to expand?**
  `heapq.heappop(frontier)` pops the node with minimum evaluation value $f(n) = g(n) + h(n)$.
- **(c) Where is the heuristic calculated?**
  When evaluating candidate child nodes before insertion into the priority queue (`heuristic_fn(next_state, goal)`).
- **(d) Does the program explicitly calculate $f(n) = g(n) + h(n)$?**
  Yes: `f_score = new_g + h`.
- **(e) How does the program prevent unnecessary repeated exploration?**
  By maintaining a `visited` set (closed set) and skipping states that have already been expanded, alongside a `g_scores` dictionary that only re-queues when a strictly cheaper path is discovered.

---

## 6. Task 5: BFS vs. $A^*$ Comparison

### Original Warehouse (Constrained Single-Corridor Labyrinth)
On the original 9x17 maze, there are exactly 64 non-wall cells forming a winding corridor with no open branching alternatives:

| Measure | BFS | $A^*$ (Manhattan) |
| :--- | :--- | :--- |
| **Solution found** | `True` | `True` |
| **Path length** | `40` | `40` |
| **States expanded** | `64` | `64` |

### Open Branching Warehouse (Alternative Open Corridors)
When alternative branches exist (e.g. open warehouse layout):

| Measure | BFS | $A^*$ (Manhattan) |
| :--- | :--- | :--- |
| **Solution found** | `True` | `True` |
| **Path length** | `20` | `20` |
| **States expanded** | `59` | **`23`** |

### Analysis
- **(a) Did both algorithms find a solution?** Yes.
- **(b) Did they find paths of the same length?** Yes, both found the optimal shortest path.
- **(c) Which algorithm expanded fewer states?** In mazes with multiple routes, $A^*$ expanded **$61\%$ fewer states** (23 vs 59). In single-corridor mazes with zero alternative branches, both expand all reachable cells (64).
- **(d) Why does $A^*$ expand fewer states?** Because $h(n)$ pulls the search tree towards the goal, heavily penalizing exploration into branches moving away from the destination.

---

## 7. Task 6: Heuristic Investigation

We evaluated four heuristic formulations:
1. $h(n) = 0$ (Dijkstra / Uniform Cost Search)
2. $h(n) = \text{Euclidean distance} = \sqrt{(r - r_G)^2 + (c - c_G)^2}$
3. $h(n) = \text{Manhattan distance} = |r - r_G| + |c - c_G|$
4. $h(n) = 2 \times \text{Manhattan distance}$ (Overestimating / Inadmissible)

### Results on Branching Warehouse

| Heuristic Formulation | Found? | Path Length | States Expanded | Admissible? |
| :--- | :---: | :---: | :---: | :---: |
| **$h(n) = 0$ (Dijkstra)** | `True` | 20 | 59 | Yes (trivial lower bound) |
| **$h(n) = \text{Euclidean}$** | `True` | 20 | 23 | Yes ($h(n) \le h^*(n)$) |
| **$h(n) = \text{Manhattan}$** | `True` | 20 | 23 | Yes (Exact distance on 4-way grid without obstacles) |
| **$h(n) = 2 \times \text{Manhattan}$** | `True` | 20 | **21** | No (Overestimates true cost) |

### Key Insights
- **Admissibility & Optimality**: Manhattan distance is admissible because without obstacles, a 4-connected grid requires at least $|r_1 - r_2| + |c_1 - c_2|$ steps. Obstacles only increase path cost, ensuring $h(n) \le h^*(n)$.
- **Overestimation ($2 \times \text{Manhattan}$)**: Inflating the heuristic makes $A^*$ more greedy. It reduces state expansions (21 states) but sacrifices the theoretical guarantee of optimality in arbitrary graphs.

---

## 8. Task 7 & Section 6: Reflections

### LLM Evaluation
1. **What parts were immediately correct?** Standard min-heap priority queue operations and coordinate offsets.
2. **Bugs / Pitfalls discovered**: Tie-breaking error when two heap items have identical $f$-scores (resolved by adding a monotonic integer counter to heap tuples).
3. **Most useful tests**: The trivial adjacent test and the blocked wall test.
4. **Could you trust the code without testing?** No. Subtle bugs (e.g. evaluating $h(n)$ after popping rather than when computing $f(n)$, or improper closed-set checking) can silently destroy $A^*$ optimality or create infinite loops.

### Final Reflection
1. **Why formulate the search problem first?** Defining $(S, A, T, s_0, G, c)$ prevents conflating the environment model with search control logic.
2. **In what sense is $A^*$ informed?** It incorporates domain-specific goal-distance estimates $h(n)$ rather than blindly exploring blind step horizons.
3. **Why does heuristic choice matter?** A dominated heuristic ($h_1 < h_2$) expands strictly more or equal states. An inadmissible heuristic risks finding sub-optimal paths.
4. **Central takeaway**: $\text{AI Science} \longrightarrow \text{AI Engineering}$. The scientific theory specifies optimality and admissibility; engineering guarantees correct data structures, edge-case handling, and validated execution.
