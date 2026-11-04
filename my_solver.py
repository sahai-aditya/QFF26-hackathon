import math
import random
from fractions import Fraction
from itertools import combinations

from qiskit import QuantumCircuit

from silentgroup import oracle_gate


MAX_ORDER_BITS = 60
ORDER_BITS = 2 * MAX_ORDER_BITS + 10
ORDER_MAX_REL_ERR = 0.02

LOG_MARGIN = 8
WINDOW = 12

VERIFY_SHOTS = 16
VERIFY_MAX_ONES = 0
VERIFY_UNSURE_ONES = 3
DEFAULT_BUDGET = 80_000
MAX_ORDER_SHOTS = 6
MAX_LOG_SHOTS = 8
MAX_AMBIGUITY = 4
KEYS = ("A", "B", "C")


class Budget:
    """Tracks how many queries we may still spend, locally and in the judge."""

    def __init__(self, oracle):
        self.oracle = oracle
        left = oracle.budget_left()
        self.limit = oracle.queries_used + (left if left is not None else DEFAULT_BUDGET)

    def left(self) -> int:
        return self.limit - self.oracle.queries_used

    def can_afford(self, cost: int, reserve: int = 0) -> bool:
        return cost + reserve <= self.left()


def build_phase_circuit(spec):
    """Read each element's turn to t digits, all in one shot (one qubit per element).
    Per digit: H -> element^(2^(k-1)) -> undo known digits -> H -> measure."""
    total_bits = sum(s[1] for s in spec)
    qc = QuantumCircuit(len(spec), total_bits)
    offset = 0

    for q, s in enumerate(spec):
        element, t = s[0], s[1]
        base = s[2] if len(s) > 2 else 1

        for k in range(t, 0, -1):
            if k != t:
                qc.reset(q)

            qc.h(q)
            qc.append(oracle_gate(element, base << (k - 1)), [q])

            for l in range(k + 1, min(t, k + WINDOW) + 1):
                with qc.if_test((qc.clbits[offset + l - 1], 1)):
                    qc.p(-2 * math.pi / (1 << (l - k + 1)), q)

            qc.h(q)
            qc.measure(q, offset + k - 1)

        offset += t

    return qc


def parse_phases(key: str, spec):
    """Turn a counts key into {element: (K, t)} meaning phase ~ K / 2^t."""
    bits = key[::-1]
    out, offset = {}, 0

    for s in spec:
        element, t = s[0], s[1]
        K = 0

        for k in range(1, t + 1):
            if bits[offset + k - 1] == "1":
                K |= 1 << (t - k)

        out[element] = (K, t)
        offset += t

    return out


class PhaseReader:
    """Caches circuits and runs one shot at a time."""

    def __init__(self, oracle, budget):
        self.oracle, self.budget = oracle, budget
        self.cache = {}

    def circuit(self, spec):
        key = tuple(spec)
        if key not in self.cache:
            qc = build_phase_circuit(spec)
            self.cache[key] = (qc, self.oracle.cost(qc, 1))

        return self.cache[key]

    def cost(self, spec):
        return self.circuit(spec)[1]

    def read(self, spec):
        """One shot. Returns {element: (K, t)}."""
        return self.read_many(spec, 1)[0]

    def read_many(self, spec, shots):
        """Several shots in one run. Returns a list of {element: (K, t)}."""
        qc, _ = self.circuit(spec)
        counts = self.oracle.run(qc, shots=shots)
        return [parse_phases(key, spec) for key, c in counts.items() for _ in range(c)]


def phase_cost(t: int):
    """Queries for reading one element's phase to t bits: 1 + 2 + ... + t."""
    return t * (t + 1) // 2


def identity_circuit(terms):
    """One qubit: H -> all oracle gates -> H -> measure. Turns add up, so if the
    product is e the qubit always reads 0; otherwise 1s appear (~half the shots)."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)

    for element, power in terms:
        if power != 0:
            qc.append(oracle_gate(element, power), [0])

    qc.h(0)
    qc.measure(0, 0)
    return qc


def identity_cost(terms) -> int:
    return VERIFY_SHOTS * sum(max(1, abs(p).bit_length()) for _, p in terms if p != 0)


class IdentityTester:
    def __init__(self, oracle, budget):
        self.oracle, self.budget = oracle, budget
        self.cache = {}

    def __call__(self, terms):
        """True / False, or None if we cannot afford the test."""
        terms = tuple((e, p) for e, p in terms if p != 0)

        if terms in self.cache:
            return self.cache[terms]

        if not terms:
            return True

        if not self.budget.can_afford(identity_cost(terms)):
            return None

        qc = identity_circuit(terms)
        ones = self.oracle.run(qc, shots=VERIFY_SHOTS).get("1", 0)

        if ones <= VERIFY_MAX_ONES:
            result = True                 # identity reads ~0.2% ones (noise)

        elif ones > VERIFY_UNSURE_ONES or not self.budget.can_afford(identity_cost(terms)):
            result = False                # non-identity reads ~50% ones

        else:
            # Borderline (1-3 ones): rare for both cases. Take 16 more shots.
            ones += self.oracle.run(qc, shots=VERIFY_SHOTS).get("1", 0)
            result = ones <= VERIFY_UNSURE_ONES

        self.cache[terms] = result
        return result


def is_probable_prime(n: int):
    """Deterministic Miller-Rabin for n < 3.3e24 (enough for 64-bit numbers)."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def pollard_brent(n: int) -> int:
    """Return a non-trivial factor of composite n."""
    if n % 2 == 0:
        return 2
    rng = random.Random(n)
    while True:
        y, c, m = rng.randrange(1, n), rng.randrange(1, n), 128
        g = r = q = 1
        x = ys = y
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g


def factorize(n: int) -> dict:
    """{prime: exponent}."""
    factors = {}
    for p in (2, 3, 5, 7, 11, 13):
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    stack = [n] if n > 1 else []
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_probable_prime(m):
            factors[m] = factors.get(m, 0) + 1
            continue
        d = pollard_brent(m)
        stack += [d, m // d]
    return factors


def valuation(x: int, p: int) -> int:
    v = 0
    while x and x % p == 0:
        x //= p
        v += 1
    return v


def crt_pair(r1, m1, r2, m2):
    """Solve x = r1 (mod m1), x = r2 (mod m2) for any moduli. None if inconsistent."""
    g = math.gcd(m1, m2)
    if (r2 - r1) % g:
        return None

    l = m1 // g * m2
    t = ((r2 - r1) // g) * pow(m1 // g, -1, m2 // g) % (m2 // g) if m2 // g > 1 else 0
    return (r1 + m1 * t) % l, l


def denominator_from_phase(K: int, t: int):
    """Continued fractions: K/2^t ≈ j/n -> denominator r (divides n).
    Returns None if the reading isn't close to any fraction (noise-corrupted)."""
    N = 1 << MAX_ORDER_BITS
    f = Fraction(K, 1 << t).limit_denominator(N)
    dist = abs(Fraction(K, 1 << t) - f) * f.denominator * N

    if dist > ORDER_MAX_REL_ERR:
        return None

    return f.denominator


def order_candidates(rs):
    """Candidate multiples of ord(g): lcms of subsets of shot denominators that stay
    <= 2^60 (a noisy shot pushes the lcm above that). Most-supported first."""
    best = {}
    idx = range(len(rs))
    for size in range(len(rs), 0, -1):
        for combo in combinations(idx, size):
            used = [rs[i] for i in combo]
            M = math.lcm(*used)

            if M > (1 << MAX_ORDER_BITS):
                continue

            if M not in best or len(best[M][1]) < size:
                best[M] = (M, used)

    ranked = []
    for M, used in best.values():
        ranked.append((len(used), M, used))

    ranked.sort(reverse=True)
    result = []
    for count, M, used in ranked:
        result.append((M, used))

    return result


REFINE_BITS = 24
REFINE_SHOTS = 3
REFINE_ATTEMPTS = 3


def refine_order(M, reader, budget, reserve):
    """g^M != e, but M is probably n / d for a small d. The element g^M has order
    exactly d, so a short phase estimation on g^M (powers M * 2^k) finds d.
    Returns a new candidate M * d', or None."""
    spec = [("g", REFINE_BITS, M)]
    d = 1

    for _ in range(REFINE_ATTEMPTS):     # a round can see phase 0 every time; retry
        if not budget.can_afford(reader.cost(spec) * REFINE_SHOTS, reserve):
            break

        for phases in reader.read_many(spec, REFINE_SHOTS):
            K, t = phases["g"]
            r = Fraction(K, 1 << t).limit_denominator(1 << 10).denominator
            d = math.lcm(d, r)

        if d > 1:
            break

    if d == 1 or M * d > (1 << MAX_ORDER_BITS):
        return None

    return M * d


def trim_powers_of_two(M, is_identity):
    """A bit flip at position k shifts a phase by exactly 2^-k, which can make a
    corrupted reading's denominator n * 2^s: too big by a power of 2 only. Check
    g^(M/2); if it is still e, binary-search the excess (log2 tests, not linear)."""
    v = valuation(M, 2)
    if v == 0 or not is_identity([("g", M // 2)]):
        return M

    lo, hi = 1, v                        # g^(M/2^lo) = e is known; find the largest
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if is_identity([("g", M >> mid)]):
            lo = mid

        else:
            hi = mid - 1

    return M // 2**lo


def trim_order(M, used, is_identity):
    """Shrink a verified multiple M of ord(g) to the true order: drop excess 2s, then
    drop any prime p backed by fewer than 2 shots while g^(M/p) is still e."""
    M = trim_powers_of_two(M, is_identity)
    if len(used) < 2:
        return M

    for p, e in factorize(M).items():
        if p == 2:
            continue                     # already handled

        support = sum(1 for r in used if valuation(r, p) == e)
        if support >= 2:
            continue

        while M % p == 0:
            ok = is_identity([("g", M // p)])
            if not ok:
                break

            M //= p

    return M


def round_to_multiple(K: int, t: int, n: int):
    """Phase K/2^t ~ m/n: return (m mod n, distance from the nearest multiple)."""
    x = Fraction(K * n, 1 << t)
    m = round(x)
    return m % n, abs(float(x - m))


def log_candidates(pairs, n):
    """pairs = (j, m) per shot, with m = x*j (mod n) if X = g^x. Solve each shot,
    combine shots (CRT), and rank candidate x by how many shots agree. Best first."""
    cosets = []
    for j, m in pairs:
        d = math.gcd(j, n)
        if m % d:
            cosets.append(None)       # this shot says X is not a power of g (or noise)
            continue

        mod = n // d
        x = (m // d) * pow(j // d, -1, mod) % mod if mod > 1 else 0
        cosets.append((x, mod))

    valid = [c for c in cosets if c is not None]
    candidates = set()
    for size in range(1, min(len(valid), 4) + 1):
        for combo in combinations(valid, size):
            r, mod = 0, 1
            for (x, m2) in combo:
                res = crt_pair(r, mod, x, m2)

                if res is None:
                    break

                r, mod = res

            else:
                k = n // mod
                if k <= MAX_AMBIGUITY:
                    for i in range(k):
                        candidates.add((r + i * mod) % n if n > 1 else 0)

    scored = []
    for x in candidates:
        score = sum(1 for j, m in pairs if (x * j - m) % n == 0)
        scored.append((score, x))

    scored.sort(reverse=True)
    return scored


def solve(oracle) -> dict:
    budget = Budget(oracle)
    reader = PhaseReader(oracle, budget)
    is_identity = IdentityTester(oracle, budget)

    shots = []   # each: {element: (K, t)} from one shot

    # ---------------- Step 1: order of g ----------------
    # Read g alone to ~126 bits: enough for continued fractions with n <= 2^60.
    # (A, B, C are read later, at only the precision the actual n needs.)
    order_spec = [("g", ORDER_BITS)]
    verify_reserve = 3 * identity_cost([("g", 1 << 59)])

    rs, n, tested, refined = [], None, set(), set()
    best_M = 1
    order_shots = 0
    while order_shots < MAX_ORDER_SHOTS:
        if not budget.can_afford(reader.cost(order_spec), verify_reserve):
            break

        phases = reader.read(order_spec)
        order_shots += 1
        r = denominator_from_phase(*phases["g"])
        if r is None:
            continue        # corrupted reading: discard it and take another shot

        rs.append(r)

        # Test the best (most-supported) untested candidates; at most 2 per shot,
        # since a failure usually just means we need another shot.
        cands = [c for c in order_candidates(rs) if c[0] not in tested]
        if cands:
            best_M = cands[0][0]

        top = len(cands[0][1]) if cands else 0
        for i, (M, used) in enumerate([c for c in cands if len(c[1]) == top][:2]):
            tested.add(M)
            ok = is_identity([("g", M)])
            if ok:
                n = trim_order(M, used, is_identity)
                break

            if ok is None:
                break

            # g^M != e. If M = n/d for a small d, find d cheaply from g^M itself.
            if i == 0 and M not in refined:
                refined.add(M)
                M2 = refine_order(M, reader, budget, verify_reserve)
                if M2 is not None and M2 not in tested:
                    tested.add(M2)
                    if is_identity([("g", M2)]):
                        n = trim_order(M2, used + [M2], is_identity)
                        break

        if n is not None:
            break

    if n is None:
        # Out of budget or shots: report our best guess for the order.
        return {"order": best_M, "secret": "none"}

    # ---------------- Step 2: discrete logs of A, B, C ----------------
    logs = {}
    pending = list(KEYS)
    t_log = n.bit_length() + LOG_MARGIN

    def collect_pairs(element):
        pairs = []
        for ph in shots:
            if element not in ph:
                continue

            j, gap = round_to_multiple(*ph["g"], n)
            if gap > 0.3:          # g's reading in this shot is unreliable
                continue

            m, _ = round_to_multiple(*ph[element], n)
            pairs.append((j, m))

        return pairs

    while pending:
        forged = False
        for X in list(pending):
            pairs = collect_pairs(X)
            scored = log_candidates(pairs, n)
            good = [(s, x) for s, x in scored if s >= 2 and s >= 0.5 * len(pairs)]
            if good:
                for _, x in good[:MAX_AMBIGUITY]:
                    ok = is_identity([(X, 1), ("g", -x)])
                    if ok:
                        logs[X] = x
                        pending.remove(X)
                        break

                else:
                    forged = True      # the best-supported answers all failed the check

            elif len(pairs) >= MAX_LOG_SHOTS:
                forged = True          # many shots, still no consistent answer

            if forged:
                break

        if forged:
            return {"order": n, "secret": "none"}

        if not pending:
            break

        # Need more data: one more shot reading g and only the still-unsolved elements.
        spec = [("g", t_log)] + [(X, t_log) for X in pending]
        reserve = len(pending) * identity_cost([("g", n)])
        if not budget.can_afford(reader.cost(spec), reserve):
            # Last resort: test the best single-shot candidates we have.
            for X in list(pending):
                scored = log_candidates(collect_pairs(X), n)
                for _, x in scored[:1]:
                    if is_identity([(X, 1), ("g", -x)]):
                        logs[X] = x
                        pending.remove(X)

            if pending:
                return {"order": n, "secret": "none"}

            break
        shots.append(reader.read(spec))

    secret = (logs["A"] * logs["B"] * logs["C"]) % n if n > 1 else 0
    return {"order": n, "secret": secret}
