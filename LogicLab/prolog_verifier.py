"""
Prolog Logical Verification Simulator
Demonstrates Tasks 6, 7, and 8 of the Logic Lab in Python without requiring external SWI-Prolog installation.
"""

class PrologKnowledgeBase:
    def __init__(self):
        # Facts: connected(X, Y)
        self.facts_connected = {
            ('a', 'b'),
            ('b', 'a'),
            ('b', 'c'),
            ('c', 'b')
        }
        # Task 8 facts
        self.facts = {"wet_road"}

    # Rule: can_move(X, Y) :- connected(X, Y)
    def can_move(self, x: str, y: str) -> bool:
        return (x, y) in self.facts_connected

    # Rule: valid_move(X, Y) :- connected(X, Y)
    def valid_move(self, x: str, y: str) -> bool:
        return (x, y) in self.facts_connected

    # Task 8 rules:
    # slippery :- wet_road.
    # reduce_speed :- slippery.
    def is_slippery(self) -> bool:
        return "wet_road" in self.facts

    def should_reduce_speed(self) -> bool:
        return self.is_slippery()


def run_prolog_verification():
    print("=" * 60)
    print("PROLOG LOGICAL VERIFICATION TEST SUITE (Section 7)")
    print("=" * 60)

    kb = PrologKnowledgeBase()

    # Task 6: can_move queries
    print("\n[Task 6 Queries]")
    q1 = kb.can_move('a', 'b')
    q2 = kb.can_move('a', 'c')
    print(f"?- can_move(a, b).  -->  {q1} (Direct edge exists)")
    print(f"?- can_move(a, c).  -->  {q2} (No direct edge exists)")

    # Task 7: Plan action verification
    print("\n[Task 7 Queries: Verifying Proposed Plan Moves]")
    moves = [('a', 'b'), ('b', 'c'), ('a', 'c')]
    for src, dst in moves:
        res = kb.valid_move(src, dst)
        status = "SUCCEEDED" if res else "FAILED"
        print(f"?- valid_move({src}, {dst}).  -->  {res} ({status})")

    # Task 8: Chained logical inference
    print("\n[Task 8: Logical Reasoning Chain]")
    res_speed = kb.should_reduce_speed()
    print("Fact  : wet_road.")
    print("Rule 1: slippery :- wet_road.")
    print("Rule 2: reduce_speed :- slippery.")
    print(f"Query : ?- reduce_speed.  -->  {res_speed}")
    print("Proof : wet_road => slippery => reduce_speed")
    print("=" * 60)


if __name__ == "__main__":
    run_prolog_verification()
