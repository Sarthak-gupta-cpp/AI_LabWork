"""
Warehouse Navigation Problem - Goal-Based Agent
Artificial Intelligence Laboratory: Agents Lab

This module implements a goal-based intelligent agent that navigates a 2D warehouse grid
from a start position (S) to a goal position (G) avoiding obstacle walls (#).
"""

from collections import deque
import heapq
from typing import List, Tuple, Optional, Dict, Set


# The default warehouse map specified in the laboratory handout
DEFAULT_WAREHOUSE_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################"
]


class WarehouseEnvironment:
    """
    Represents the 2D grid warehouse environment.
    Coordinates are defined as (row, col) with (0, 0) at the top-left.
    """
    def __init__(self, grid_map: List[str]):
        self.grid = [list(row) for row in grid_map]
        self.height = len(self.grid)
        self.width = len(self.grid[0]) if self.height > 0 else 0
        self.start = self._find_symbol('S')
        self.goal = self._find_symbol('G')

        if not self.start:
            raise ValueError("Start position 'S' not found in the grid map.")
        if not self.goal:
            raise ValueError("Goal position 'G' not found in the grid map.")

    def _find_symbol(self, symbol: str) -> Optional[Tuple[int, int]]:
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == symbol:
                    return (r, c)
        return None

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid[r][c] != '#'
        return False

    def get_actions(self, pos: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Returns valid actions and resulting successor positions from the given position.
        Possible actions: Up, Down, Left, Right.
        """
        r, c = pos
        candidates = [
            ("Up", (r - 1, c)),
            ("Down", (r + 1, c)),
            ("Left", (r, c - 1)),
            ("Right", (r, c + 1)),
        ]
        valid_transitions = []
        for action, next_pos in candidates:
            if self.is_valid_position(next_pos):
                valid_transitions.append((action, next_pos))
        return valid_transitions

    def render_path(self, path: List[Tuple[int, int]]) -> str:
        """
        Renders the grid with the path marked with '*'.
        """
        display_grid = [row[:] for row in self.grid]
        path_set = set(path)
        for r, c in path_set:
            if (r, c) != self.start and (r, c) != self.goal:
                display_grid[r][c] = '*'
        return "\n".join("".join(row) for row in display_grid)


class GoalBasedAgent:
    """
    A goal-based agent that plans a sequence of actions to reach its destination.
    Maintains an explicit representation of its goal and state.
    """
    def __init__(self, env: WarehouseEnvironment, algorithm: str = "bfs"):
        self.env = env
        self.start = env.start
        self.goal = env.goal
        self.algorithm = algorithm.lower()

    def _manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def plan_path(self) -> Dict:
        """
        Executes search to find a collision-free path from start to goal.
        Supports both BFS (uninformed, optimal for unit costs) and A* (informed).
        """
        if self.algorithm == "astar":
            return self._plan_astar()
        return self._plan_bfs()

    def _plan_bfs(self) -> Dict:
        start = self.start
        goal = self.goal
        frontier = deque([(start, [start], [])])  # (current_pos, path_coords, actions)
        visited: Set[Tuple[int, int]] = {start}
        states_expanded = 0

        while frontier:
            current, path, actions = frontier.popleft()
            states_expanded += 1

            if current == goal:
                return {
                    "success": True,
                    "algorithm": "BFS (Breadth-First Search)",
                    "path": path,
                    "actions": actions,
                    "path_length": len(actions),
                    "states_expanded": states_expanded
                }

            for action, successor in self.env.get_actions(current):
                if successor not in visited:
                    visited.add(successor)
                    frontier.append((successor, path + [successor], actions + [action]))

        return {
            "success": False,
            "algorithm": "BFS (Breadth-First Search)",
            "path": [],
            "actions": [],
            "path_length": 0,
            "states_expanded": states_expanded
        }

    def _plan_astar(self) -> Dict:
        start = self.start
        goal = self.goal
        # Priority queue entries: (f_score, g_score, count, current_pos, path, actions)
        counter = 0
        initial_h = self._manhattan_distance(start, goal)
        frontier = [(initial_h, 0, counter, start, [start], [])]
        g_scores = {start: 0}
        states_expanded = 0
        visited: Set[Tuple[int, int]] = set()

        while frontier:
            f, g, _, current, path, actions = heapq.heappop(frontier)

            if current in visited:
                continue
            visited.add(current)
            states_expanded += 1

            if current == goal:
                return {
                    "success": True,
                    "algorithm": "A* Search (Manhattan Heuristic)",
                    "path": path,
                    "actions": actions,
                    "path_length": len(actions),
                    "states_expanded": states_expanded
                }

            for action, successor in self.env.get_actions(current):
                tentative_g = g + 1
                if successor not in g_scores or tentative_g < g_scores[successor]:
                    g_scores[successor] = tentative_g
                    h = self._manhattan_distance(successor, goal)
                    f_score = tentative_g + h
                    counter += 1
                    heapq.heappush(frontier, (f_score, tentative_g, counter, successor, path + [successor], actions + [action]))

        return {
            "success": False,
            "algorithm": "A* Search (Manhattan Heuristic)",
            "path": [],
            "actions": [],
            "path_length": 0,
            "states_expanded": states_expanded
        }


def run_experiment():
    print("=" * 60)
    print("ARTIFICIAL INTELLIGENCE - AGENTS LAB")
    print("Warehouse Navigation: Goal-Based Agent")
    print("=" * 60)

    env = WarehouseEnvironment(DEFAULT_WAREHOUSE_MAP)
    print(f"\nWarehouse Dimensions: {env.height} rows x {env.width} columns")
    print(f"Start Position (S): {env.start}")
    print(f"Goal Position  (G): {env.goal}\n")

    # Run with BFS
    agent_bfs = GoalBasedAgent(env, algorithm="bfs")
    result_bfs = agent_bfs.plan_path()

    print(f"--- Algorithm: {result_bfs['algorithm']} ---")
    if result_bfs["success"]:
        print(f"Path Found: YES")
        print(f"Path Length (steps): {result_bfs['path_length']}")
        print(f"States Expanded: {result_bfs['states_expanded']}")
        print(f"Actions ({len(result_bfs['actions'])}): {', '.join(result_bfs['actions'])}")
        print("\nMap with Traversed Path (*):")
        print(env.render_path(result_bfs["path"]))
    else:
        print("No collision-free path exists from S to G.")

    print("\n" + "-" * 60)

    # Run with A*
    agent_astar = GoalBasedAgent(env, algorithm="astar")
    result_astar = agent_astar.plan_path()

    print(f"--- Algorithm: {result_astar['algorithm']} ---")
    if result_astar["success"]:
        print(f"Path Found: YES")
        print(f"Path Length (steps): {result_astar['path_length']}")
        print(f"States Expanded: {result_astar['states_expanded']}")
        print(f"Actions ({len(result_astar['actions'])}): {', '.join(result_astar['actions'])}")
        print("\nMap with Traversed Path (*):")
        print(env.render_path(result_astar["path"]))
    else:
        print("No collision-free path exists from S to G.")

    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
