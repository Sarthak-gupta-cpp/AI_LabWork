"""
Artificial Intelligence - Logic Laboratory
Logical Reasoning for Planning: STRIPS-Style Planning Agent

This module implements:
1. Propositional State Representation
2. Action Schema with positive/negative preconditions and positive/negative effects
3. Breadth-First Search (BFS) Planning Engine (Logic + Search = Planning)
4. Comprehensive Test Suite:
   - Test A: Solvable Problem (Standard Warehouse Cargo Delivery)
   - Test B: Impossible Problem (Missing PickUp Action -> "No plan found")
   - Test C: Irrelevant / Distractor Actions (Moving without picking up cargo)
5. Plan Verification and State Transition Validation
"""

from collections import deque
from typing import Set, List, Dict, Optional, FrozenSet, Tuple


class Action:
    """
    Represents a planning action with:
    - name: identifier string
    - pos_preconds: facts that MUST be present in the state
    - neg_preconds: facts that MUST NOT be present in the state
    - pos_effects: facts added to the successor state
    - neg_effects: facts removed from the successor state
    """
    def __init__(
        self,
        name: str,
        pos_preconds: Set[str],
        neg_preconds: Optional[Set[str]] = None,
        pos_effects: Optional[Set[str]] = None,
        neg_effects: Optional[Set[str]] = None
    ):
        self.name = name
        self.pos_preconds = set(pos_preconds)
        self.neg_preconds = set(neg_preconds) if neg_preconds else set()
        self.pos_effects = set(pos_effects) if pos_effects else set()
        self.neg_effects = set(neg_effects) if neg_effects else set()

    def is_applicable(self, state: FrozenSet[str]) -> bool:
        """
        Action applicability test: S |= Preconditions(a)
        All positive preconditions must be in state, and no negative preconditions.
        """
        # S satisfies positive preconditions
        if not self.pos_preconds.issubset(state):
            return False
        # S satisfies negative preconditions
        if any(neg in state for neg in self.neg_preconds):
            return False
        return True

    def apply(self, state: FrozenSet[str]) -> FrozenSet[str]:
        """
        Computes successor state: S' = (S \\ neg_effects) U pos_effects
        """
        new_state = set(state)
        new_state.difference_update(self.neg_effects)
        new_state.update(self.pos_effects)
        return frozenset(new_state)

    def __repr__(self) -> str:
        return self.name


class PlanningProblem:
    """Defines a planning instance: (I, A, G)."""
    def __init__(self, initial_state: Set[str], goal_state: Set[str], actions: List[Action]):
        self.initial_state = frozenset(initial_state)
        self.goal_state = set(goal_state)
        self.actions = actions

    def is_goal_satisfied(self, state: FrozenSet[str]) -> bool:
        """Checks whether state entails the goal: S |= G."""
        return self.goal_state.issubset(state)


class BFSPlanner:
    """Breadth-First Search planner over states."""
    def __init__(self, problem: PlanningProblem):
        self.problem = problem

    def solve(self) -> Dict:
        initial = self.problem.initial_state
        if self.problem.is_goal_satisfied(initial):
            return {
                "success": True,
                "plan": [],
                "states": [initial],
                "message": "Goal already satisfied in initial state."
            }

        # Queue elements: (current_state, plan_actions, state_history)
        queue = deque([(initial, [], [initial])])
        visited: Set[FrozenSet[str]] = {initial}

        while queue:
            current_state, plan, states = queue.popleft()

            # Iterate over all available actions and check applicability
            for action in self.problem.actions:
                if action.is_applicable(current_state):
                    next_state = action.apply(current_state)

                    if self.problem.is_goal_satisfied(next_state):
                        return {
                            "success": True,
                            "plan": plan + [action],
                            "states": states + [next_state],
                            "message": f"Plan found with {len(plan) + 1} steps."
                        }

                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, plan + [action], states + [next_state]))

        return {
            "success": False,
            "plan": [],
            "states": [],
            "message": "No plan found"
        }


def build_warehouse_actions(allow_pickup: bool = True) -> List[Action]:
    """Generates the grounded actions for locations A, B, C."""
    actions = []
    locations = ["A", "B", "C"]
    connections = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]

    # 1. Move actions
    for src, dst in connections:
        actions.append(
            Action(
                name=f"Move({src}, {dst})",
                pos_preconds={f"At(Robot, {src})"},
                neg_effects={f"At(Robot, {src})"},
                pos_effects={f"At(Robot, {dst})"}
            )
        )

    # 2. PickUp actions
    if allow_pickup:
        for loc in locations:
            actions.append(
                Action(
                    name=f"PickUp(Package, {loc})",
                    pos_preconds={f"At(Robot, {loc})", f"At(Package, {loc})"},
                    neg_effects={f"At(Package, {loc})"},
                    pos_effects={"Holding(Package)"}
                )
            )

    # 3. Drop actions
    for loc in locations:
        actions.append(
            Action(
                name=f"Drop(Package, {loc})",
                pos_preconds={f"At(Robot, {loc})", "Holding(Package)"},
                neg_effects={"Holding(Package)"},
                pos_effects={f"At(Package, {loc})"}
            )
        )

    return actions


def verify_plan(initial_state: FrozenSet[str], plan: List[Action], goal: Set[str]) -> Tuple[bool, str]:
    """Independent verification of plan execution."""
    current = initial_state
    for step_idx, action in enumerate(plan, start=1):
        if not action.is_applicable(current):
            return False, f"Step {step_idx}: Preconditions for {action.name} violated in state: {set(current)}"
        current = action.apply(current)

    if not goal.issubset(current):
        return False, f"Plan terminated, but goal {goal} is not satisfied in final state: {set(current)}"

    return True, "Plan independently verified: all preconditions and goals satisfied."


def run_tests():
    print("=" * 70)
    print("ARTIFICIAL INTELLIGENCE - LOGIC PLANNING LAB")
    print("=" * 70)

    # -------------------------------------------------------------
    # TEST A: SOLVABLE WAREHOUSE PROBLEM
    # -------------------------------------------------------------
    print("\n[TEST A: SOLVABLE PROBLEM]")
    initial_A = {"At(Robot, A)", "At(Package, A)"}
    goal_A = {"At(Package, C)"}
    actions_A = build_warehouse_actions(allow_pickup=True)
    problem_A = PlanningProblem(initial_A, goal_A, actions_A)
    planner_A = BFSPlanner(problem_A)
    result_A = planner_A.solve()

    print(f"Initial State : {sorted(list(initial_A))}")
    print(f"Goal State    : {sorted(list(goal_A))}")
    print(f"Plan Found?   : {result_A['success']}")
    if result_A["success"]:
        print(f"Actions in Plan ({len(result_A['plan'])}):")
        for i, a in enumerate(result_A['plan'], 1):
            print(f"  Step {i}: {a.name}")
        print("\nState Progression:")
        for idx, s in enumerate(result_A['states']):
            print(f"  S{idx}: {sorted(list(s))}")

        valid, v_msg = verify_plan(frozenset(initial_A), result_A['plan'], goal_A)
        print(f"Independent Plan Validation: {'VALID' if valid else 'INVALID'} ({v_msg})")

    # -------------------------------------------------------------
    # TEST B: IMPOSSIBLE PROBLEM (No PickUp Action)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[TEST B: IMPOSSIBLE PROBLEM (PickUp Action Removed)]")
    initial_B = {"At(Robot, A)", "At(Package, A)"}
    goal_B = {"At(Package, C)"}
    actions_B = build_warehouse_actions(allow_pickup=False)  # Remove PickUp
    problem_B = PlanningProblem(initial_B, goal_B, actions_B)
    planner_B = BFSPlanner(problem_B)
    result_B = planner_B.solve()

    print(f"Initial State : {sorted(list(initial_B))}")
    print(f"Goal State    : {sorted(list(goal_B))}")
    print(f"Plan Found?   : {result_B['success']}")
    print(f"Planner Output: {result_B['message']}")

    # -------------------------------------------------------------
    # TEST C: IRRELEVANT ACTIONS (Robot moves, cargo untouched)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[TEST C: IRRELEVANT / DISTRACTOR ACTIONS]")
    # Scenario: Robot is at B, Package is at A. Robot moves to C without package.
    initial_C = {"At(Robot, B)", "At(Package, A)"}
    goal_C = {"At(Package, C)"}
    problem_C = PlanningProblem(initial_C, goal_C, actions_A)
    planner_C = BFSPlanner(problem_C)
    result_C = planner_C.solve()

    print(f"Initial State : {sorted(list(initial_C))}")
    print(f"Goal State    : {sorted(list(goal_C))}")
    print(f"Plan Found?   : {result_C['success']}")
    if result_C["success"]:
        print(f"Plan Steps ({len(result_C['plan'])}):")
        for i, a in enumerate(result_C['plan'], 1):
            print(f"  Step {i}: {a.name}")
        valid, v_msg = verify_plan(frozenset(initial_C), result_C['plan'], goal_C)
        print(f"Independent Plan Validation: {'VALID' if valid else 'INVALID'}")

    print("=" * 70)


if __name__ == "__main__":
    run_tests()
