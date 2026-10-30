"""Submission template. Define solve(oracle) and submit this one file.

Rehearse locally with:  python -m silentgroup.check examples/solver_template.py
"""

from qiskit import QuantumCircuit

from silentgroup import oracle_gate


def solve(oracle) -> dict:
    # oracle.elements          -> ["A", "B", "C", "g"]
    # oracle.run(qc, shots)    -> counts, e.g. {"0": 7, "1": 3} (clbit 0 rightmost); charges the budget
    # oracle.cost(qc, shots)   -> what that run would charge, without running it
    # oracle.budget_left()     -> queries remaining
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.append(oracle_gate("g", 1), [0])
    qc.h(0)
    qc.measure(0, 0)
    counts = oracle.run(qc, shots=10)

    # ... your algorithm here ...

    return {"order": 1, "secret": "none"}  # "secret" is abc mod ord(g), or "none" if A, B or C is forged
