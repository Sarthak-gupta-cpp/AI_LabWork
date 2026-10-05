"""
Artificial Intelligence - Search Laboratory: Search and A*
Warehouse Robot Navigation

This module implements:
1. Formal Search Problem formulation
2. A* Search with customizable heuristics (Manhattan, Zero/Dijkstra, Euclidean, Weighted)
3. Breadth-First Search (BFS) as uninformed baseline
4. Test suite covering:
   - Test 1: Original Warehouse Map (Constrained Labyrinth)
   - Test 2: Trivial Case (immediate goal)
   - Test 3: Inaccessible Goal (no solution detection)
   - Test 4: Alternative Paths (shortest path optimality verification)
5. Comprehensive Comparative Benchmarks:
   - Behavior on Constrained Labyrinth Map vs. Branching Open Warehouse Map
   - Heuristic sensitivity analysis: h=0, Euclidean, Manhattan, 2*Manhattan
"""

import math
import heapq
from collections import deque
from typing import List, Tuple, Dict, Optional, Callable, Set


ORIGINAL_WAREHOUSE_MAP = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################"
]

TRIVIAL_MAP = [
    "#####",
    "#SG##",
    "#####"
]

NO_SOLUTION_MAP = [
    "#######",
    "#S....#",
    "###.###",
    "#...#G#",
    "#######"
]

ALTERNATIVE_PATHS_MAP = [
    "#######",
    "#S...G#",
    "#.###.#",
    "#.....#",
    "#######"
]

BRANCHING_WAREHOUSE_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################"
]


class GridSearchEnvironment:
    """Represents a 2D grid navigation problem."""
    def __init__(self, grid_map: List[str]):
        self.grid = [list(row) for row in grid_map]
        self.height = len(self.grid)
        self.width = len(self.grid[0]) if self.height > 0 else 0
        self.start = self._find_char('S')
        self.goal = self._find_char('G')

        if not self.start:
            raise ValueError("Start symbol 'S' missing.")
        if not self.goal:
            raise ValueError("Goal symbol 'G' missing.")

    def _find_char(self, char: str) -> Optional[Tuple[int, int]]:
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == char:
                    return (r, c)
        return None

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width and self.grid[r][c] != '#'

    def get_successors(self, pos: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int], int]]:
        """
        Returns list of (action_name, next_state, step_cost).
        Actions: Up, Down, Left, Right with step_cost = 1.
        """
        r, c = pos
        moves = [
            ("Up", (r - 1, c)),
            ("Down", (r + 1, c)),
            ("Left", (r, c - 1)),
            ("Right", (r, c + 1)),
        ]
        successors = []
        for action, next_pos in moves:
            if self.is_valid(next_pos):
                successors.append((action, next_pos, 1))
        return successors

    def render_with_path(self, path: List[Tuple[int, int]]) -> str:
        display = [row[:] for row in self.grid]
        path_set = set(path)
        for r, c in path_set:
            if (r, c) != self.start and (r, c) != self.goal:
                display[r][c] = '*'
        return "\n".join("".join(row) for row in display)


# --- Heuristic Functions ---
def heuristic_manhattan(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def heuristic_zero(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return 0.0

def heuristic_euclidean(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

def heuristic_scaled_manhattan(scale: float) -> Callable[[Tuple[int, int], Tuple[int, int]], float]:
    return lambda pos, goal: scale * (abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))


def run_astar(env: GridSearchEnvironment, heuristic_fn: Callable[[Tuple[int, int], Tuple[int, int]], float] = heuristic_manhattan) -> Dict:
    """
    Executes A* search algorithm:
    f(n) = g(n) + h(n)
    """
    start = env.start
    goal = env.goal
    counter = 0

    # Heap elements: (f_score, g_score, counter, current_state, path, actions)
    initial_h = heuristic_fn(start, goal)
    frontier = [(initial_h, 0, counter, start, [start], [])]
    g_scores = {start: 0}
    visited: Set[Tuple[int, int]] = set()
    states_expanded = 0

    while frontier:
        f, g, _, current, path, actions = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)
        states_expanded += 1

        # Goal test
        if current == goal:
            return {
                "success": True,
                "path": path,
                "actions": actions,
                "path_length": len(actions),
                "cost": g,
                "states_expanded": states_expanded
            }

        # Expansion
        for action, next_state, step_cost in env.get_successors(current):
            new_g = g + step_cost
            if next_state not in g_scores or new_g < g_scores[next_state]:
                g_scores[next_state] = new_g
                h = heuristic_fn(next_state, goal)
                f_score = new_g + h
                counter += 1
                heapq.heappush(frontier, (f_score, new_g, counter, next_state, path + [next_state], actions + [action]))

    return {
        "success": False,
        "path": [],
        "actions": [],
        "path_length": 0,
        "cost": float('inf'),
        "states_expanded": states_expanded
    }


def run_bfs(env: GridSearchEnvironment) -> Dict:
    """
    Executes Breadth-First Search (BFS) as uninformed baseline.
    """
    start = env.start
    goal = env.goal
    frontier = deque([(start, [start], [])])
    visited: Set[Tuple[int, int]] = {start}
    states_expanded = 0

    while frontier:
        current, path, actions = frontier.popleft()
        states_expanded += 1

        if current == goal:
            return {
                "success": True,
                "path": path,
                "actions": actions,
                "path_length": len(actions),
                "cost": len(actions),
                "states_expanded": states_expanded
            }

        for action, next_state, step_cost in env.get_successors(current):
            if next_state not in visited:
                visited.add(next_state)
                frontier.append((next_state, path + [next_state], actions + [action]))

    return {
        "success": False,
        "path": [],
        "actions": [],
        "path_length": 0,
        "cost": float('inf'),
        "states_expanded": states_expanded
    }


def execute_all_laboratory_tasks():
    print("=" * 75)
    print("AI LAB - SEARCH AND A* EXPERIMENT SUITE")
    print("=" * 75)

    # 1. TASK 3 TESTS
    print("\n[TASK 3: SYSTEMATIC TESTING]")
    print("-" * 55)

    # Test 1: Original Warehouse
    env_orig = GridSearchEnvironment(ORIGINAL_WAREHOUSE_MAP)
    res_orig = run_astar(env_orig, heuristic_manhattan)
    print("\n--- Test 1: Original Warehouse Map ---")
    print(f"Solution Found  : {res_orig['success']}")
    print(f"Path Length     : {res_orig['path_length']}")
    print(f"States Expanded : {res_orig['states_expanded']}")
    print(f"Path Actions ({len(res_orig['actions'])}): {' -> '.join(res_orig['actions'])}")
    print("Rendered Map:")
    print(env_orig.render_with_path(res_orig['path']))

    # Test 2: Trivial Case
    env_triv = GridSearchEnvironment(TRIVIAL_MAP)
    res_triv = run_astar(env_triv, heuristic_manhattan)
    print("\n--- Test 2: Trivial Case (Adjacent Goal) ---")
    print(f"Solution Found  : {res_triv['success']}")
    print(f"Path Length     : {res_triv['path_length']}")
    print(f"States Expanded : {res_triv['states_expanded']}")
    print(f"Actions         : {res_triv['actions']}")

    # Test 3: No Solution
    env_nosol = GridSearchEnvironment(NO_SOLUTION_MAP)
    res_nosol = run_astar(env_nosol, heuristic_manhattan)
    print("\n--- Test 3: Inaccessible Goal ---")
    print(f"Solution Found  : {res_nosol['success']}")
    print(f"States Expanded : {res_nosol['states_expanded']} (Terminated cleanly, no infinite loop)")

    # Test 4: Alternative Paths
    env_alt = GridSearchEnvironment(ALTERNATIVE_PATHS_MAP)
    res_alt = run_astar(env_alt, heuristic_manhattan)
    print("\n--- Test 4: Alternative Paths Map ---")
    print(f"Solution Found  : {res_alt['success']}")
    print(f"Path Length     : {res_alt['path_length']} (Shortest top route: 4 steps)")
    print(f"Actions         : {' -> '.join(res_alt['actions'])}")

    # 2. TASK 5: BFS VS A* COMPARISON
    print("\n" + "=" * 75)
    print("[TASK 5: A* VS BFS COMPARISON]")
    print("-" * 75)
    res_bfs_orig = run_bfs(env_orig)
    print(f"--- Map: Original Warehouse (Constrained Single-Corridor Labyrinth) ---")
    print(f"{'Measure':<20} | {'BFS':<15} | {'A* (Manhattan)':<15}")
    print("-" * 55)
    print(f"{'Solution found':<20} | {str(res_bfs_orig['success']):<15} | {str(res_orig['success']):<15}")
    print(f"{'Path length':<20} | {str(res_bfs_orig['path_length']):<15} | {str(res_orig['path_length']):<15}")
    print(f"{'States expanded':<20} | {str(res_bfs_orig['states_expanded']):<15} | {str(res_orig['states_expanded']):<15}")

    # Branching map comparison
    env_branch = GridSearchEnvironment(BRANCHING_WAREHOUSE_MAP)
    res_bfs_br = run_bfs(env_branch)
    res_astar_br = run_astar(env_branch, heuristic_manhattan)
    print(f"\n--- Map: Branching Warehouse (Alternative Open Corridors) ---")
    print(f"{'Measure':<20} | {'BFS':<15} | {'A* (Manhattan)':<15}")
    print("-" * 55)
    print(f"{'Solution found':<20} | {str(res_bfs_br['success']):<15} | {str(res_astar_br['success']):<15}")
    print(f"{'Path length':<20} | {str(res_bfs_br['path_length']):<15} | {str(res_astar_br['path_length']):<15}")
    print(f"{'States expanded':<20} | {str(res_bfs_br['states_expanded']):<15} | {str(res_astar_br['states_expanded']):<15}")

    # 3. TASK 6: HEURISTIC INVESTIGATION
    print("\n" + "=" * 75)
    print("[TASK 6: HEURISTIC INVESTIGATION]")
    print("-" * 75)

    heuristics = [
        ("h(n) = 0 (Dijkstra/UCS)", heuristic_zero),
        ("h(n) = Euclidean", heuristic_euclidean),
        ("h(n) = Manhattan", heuristic_manhattan),
        ("h(n) = 2 * Manhattan", heuristic_scaled_manhattan(2.0)),
    ]

    print("\n--- On Original Labyrinth Map ---")
    print(f"{'Heuristic Formulation':<25} | {'Found?':<8} | {'Path Length':<12} | {'States Expanded':<16}")
    print("-" * 70)
    for name, h_func in heuristics:
        r = run_astar(env_orig, h_func)
        print(f"{name:<25} | {str(r['success']):<8} | {str(r['path_length']):<12} | {str(r['states_expanded']):<16}")

    print("\n--- On Branching Warehouse Map ---")
    print(f"{'Heuristic Formulation':<25} | {'Found?':<8} | {'Path Length':<12} | {'States Expanded':<16}")
    print("-" * 70)
    for name, h_func in heuristics:
        r = run_astar(env_branch, h_func)
        print(f"{name:<25} | {str(r['success']):<8} | {str(r['path_length']):<12} | {str(r['states_expanded']):<16}")

    print("=" * 75)


if __name__ == "__main__":
    execute_all_laboratory_tasks()
