import concurrent.futures
from enum import StrEnum
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from pysat.card import CardEnc, EncType
from pysat.solvers import Solver

from alcsat.instance import ALC_OP, OP, ALCConcept, Instance
from alcsat.preprocessing import (
    bisimulation_reduction,
    decode_dataproperties,
    decode_inverses,
    determine_max_q_per_relation,
    encode_dataproperties,
    encode_inverses,
    prune_conceptnames,
    restrict_neighborhood,
    ThresholdMethod,
    color_refinement,
    extract_concept,
    merge_conj,
    deduplicate,
    simplify_conj,
)

from .fitting_el import (
    determine_relevant_symbols,
    non_empty_symbols,
)
from .structures import (
    Signature,
    Structure,
    entire_signature,
)


class FittingMode(StrEnum):
    EXACT = "exact"
    APPROX = "approx"


X = 0
Z = 1
V = 2


# Generate non-isomorphic trees of size n with at most binary outdegree
def all_trees(k: int, start: int = 0) -> list[list[tuple[int, ...]]]:
    if k == 1:
        return [[()]]

    res: list[list[tuple[int, ...]]] = []
    for i in range(1, (k - 1) // 2 + 1):
        for idx_a, a in enumerate(all_trees(i, start + 2)):
            for idx_b, b in enumerate(all_trees((k - 1) - i, start + i + 1)):
                # If a, b have the same size, skip pairs that already occured
                if i == (k - 1) - i and idx_b < idx_a:
                    continue
                # Whacky tree composition to ensure that children of binary nodes are always adjacent
                # (the start + 2 for the a trees is for the same purpose)
                res.append([(start + 1, start + 2)] + [a[0]] + [b[0]] + a[1:] + b[1:])

    for a in all_trees(k - 1, start + 1):
        res.append([(start + 1,)] + a)

    return res


def cn_types(A: Structure, sigma: Signature) -> set[frozenset[str]]:
    res: set[frozenset[str]] = set()
    # TODO: this is slow when there are many concept names and many individuals
    for i in range(A.max_ind):
        tp = frozenset(cn for cn in sigma.conceptnames if i in A.cn_ext[cn])
        res.add(tp)
    return res


class ALCSATEncoding:
    def __init__(self, instance: Instance):
        self.inst: Instance = instance
        self.solver: Solver | None = None
        self.k: int = 0
        self.vars: dict[Any, int] = {}
        self.max_var: int = 0
        self.types: set[frozenset[str]] = set()
        self.clauses: list[Iterable[int]] = []
        self.max_q_per_r: dict[str, int] = {
            r: min(q, self.inst.max_q)
            for r, q in determine_max_q_per_relation(instance).items()
        }
        self.exclude_atomic : Iterable[OP] = []

    def add_clause(self, c: Iterable[int]):
        self.clauses.append(c)

    def create_vars(self):
        d: dict[Any, int] = {}
        i = 1
        d[X, OP.TOP] = i
        d[X, OP.BOT] = i * self.k + 1
        i += 1
        for cn in self.inst.sigma.conceptnames:
            d[X, cn] = i * self.k + 1
            i += 1
        for op in self.inst.op_b():
            d[X, op] = i * self.k + 1
            i += 1
        if OP.EX in self.inst.op:
            for c in self.inst.sigma.rolenames:
                d[X, OP.EX, c] = i * self.k + 1
                i += 1
        if OP.ALL in self.inst.op:
            for c in self.inst.sigma.rolenames:
                d[X, OP.ALL, c] = i * self.k + 1
                i += 1
        if OP.LE in self.inst.op_q():
            for r in self.inst.sigma.rolenames:
                for q in range(1, self.max_q_per_r[r] + 2):
                    d[X, OP.LE, r, q] = i * self.k + 1
                    i += 1
        if OP.GE in self.inst.op_q():
            for r in self.inst.sigma.rolenames:
                for q in range(2, self.max_q_per_r[r] + 2):
                    d[X, OP.GE, r, q] = i * self.k + 1
                    i += 1
        for a in range(self.inst.A.max_ind):
            d[Z, a] = i * self.k + 1
            i += 1
        for j in range(self.k):
            d[V, 1, j] = i * self.k + 1
            i += 1
        for j in range(self.k):
            d[V, 2, j] = i * self.k + 1
            i += 1

        for tp in self.types:
            d[X, tp] = i * self.k + 1
            i += 1

        self.max_var = i * self.k + 1

        self.vars = d

    def disable_topbot(self):
        for op in self.exclude_atomic:
            for i in range(self.k):
                self.add_clause([-(self.vars[X,op]+i)])
    
    def syn_tree_encoding(self, tt: int):
        tree = all_trees(self.k)[tt]

        for i in range(self.k):
            v_vars = [self.vars[V, 1, i] + j for j in range(i + 1, self.k)] + [
                self.vars[V, 2, i] + j for j in range(i + 1, self.k - 1)
            ]

            # At most one of the y-vars
            for clause in CardEnc.atmost(lits=v_vars, encoding=EncType.pairwise):
                self.add_clause(clause)

            if len(tree[i]) == 0:
                for v in v_vars:
                    self.add_clause([-v])
                x_vars = [self.vars[X, OP.TOP] + i, self.vars[X, OP.BOT] + i] + [
                    self.vars[X, cn] + i for cn in self.inst.sigma.conceptnames
                ]
                if len(x_vars) > 0:
                    for clause in CardEnc.equals(
                        lits=x_vars, encoding=EncType.pairwise
                    ):
                        self.add_clause(clause)
                else:
                    self.add_clause([1])
                    self.add_clause([-1])
            elif len(tree[i]) == 1:
                self.add_clause([self.vars[V, 1, i] + tree[i][0]])
                x_vars = (
                    [self.vars[X, op] + i for op in {OP.NEG} if op in self.inst.op]
                    + [
                        self.vars[X, op, r] + i
                        for op in self.inst.op_r()
                        for r in self.inst.sigma.rolenames
                    ]
                    + [
                        self.vars[X, op, r, q] + i
                        for op in self.inst.op_q()
                        for r in self.inst.sigma.rolenames
                        for q in range(1, self.max_q_per_r[r] + 2)
                        if op != OP.GE or q != 1
                    ]
                )
                if len(x_vars) > 0:
                    for clause in CardEnc.equals(
                        lits=x_vars, encoding=EncType.pairwise
                    ):
                        self.add_clause(clause)
                else:
                    self.add_clause([1])
                    self.add_clause([-1])
            elif len(tree[i]) == 2:
                self.add_clause([self.vars[V, 2, i] + tree[i][0]])
                x_vars = [
                    self.vars[X, op] + i
                    for op in {OP.AND, OP.OR}.intersection(self.inst.op)
                ]
                if len(x_vars) > 0:
                    for clause in CardEnc.equals(
                        lits=x_vars, encoding=EncType.pairwise
                    ):
                        self.add_clause(clause)
                else:
                    self.add_clause([1])
                    self.add_clause([-1])
            else:
                assert False

    def symmetry_breaking(self):
        # Symmetry breaking: associativity of sqcap and sqcup
        # There is always a syntax tree where one of the successors of OP.AND is not an OP.AND
        for i in range(self.k):
            for j in range(i + 1, self.k - 1):
                if OP.AND in self.inst.op_b():
                    self.add_clause(
                        (
                            -(self.vars[X, OP.AND] + i),
                            -(self.vars[V, 2, i] + j),
                            -(self.vars[X, OP.AND] + j),
                            -(self.vars[X, OP.AND] + j + 1),
                        )
                    )
                if OP.OR in self.inst.op_b():
                    self.add_clause(
                        (
                            -(self.vars[X, OP.OR] + i),
                            -(self.vars[V, 2, i] + j),
                            -(self.vars[X, OP.OR] + j),
                            -(self.vars[X, OP.OR] + j + 1),
                        )
                    )

        # Symmetry breaking: there is always a syntax tree where OP.NEG is not nested directly under OP.ALL or OP.EX or OP.NEG
        if (
            OP.EX in self.inst.op_r()
            and OP.ALL in self.inst.op_r()
            and OP.NEG in self.inst.op_b()
        ):
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    self.add_clause(
                        (-(self.vars[V, 1, i] + j), -(self.vars[X, OP.NEG] + j))
                    )

        # Symmetry breaking: rewrites involving OP.TOP and OP.BOT?
        for i in range(self.k):
            for j in range(i + 1, self.k - 1):
                if OP.AND in self.inst.op_b():
                    self.add_clause(
                        (
                            -(self.vars[X, OP.AND] + i),
                            -(self.vars[V, 2, i] + j),
                            -(self.vars[X, OP.TOP] + j),
                        )
                    )
                    self.add_clause(
                        (
                            -(self.vars[X, OP.AND] + i),
                            -(self.vars[V, 2, i] + j),
                            -(self.vars[X, OP.TOP] + j + 1),
                        )
                    )
                    self.add_clause(
                        (
                            -(self.vars[X, OP.AND] + i),
                            -(self.vars[V, 2, i] + j),
                            -(self.vars[X, OP.BOT] + j),
                        )
                    )
                    self.add_clause(
                        (
                            -(self.vars[X, OP.AND] + i),
                            -(self.vars[V, 2, i] + j),
                            -(self.vars[X, OP.BOT] + j + 1),
                        )
                    )
                if OP.OR in self.inst.op_b():
                    self.add_clause(
                        (
                            -(self.vars[X, OP.OR] + i),
                            -(self.vars[V, 2, i] + j),
                            -(self.vars[X, OP.TOP] + j),
                        )
                    )
                    self.add_clause(
                        (
                            -(self.vars[X, OP.OR] + i),
                            -(self.vars[V, 2, i] + j),
                            -(self.vars[X, OP.TOP] + j + 1),
                        )
                    )
                    self.add_clause(
                        (
                            -(self.vars[X, OP.OR] + i),
                            -(self.vars[V, 2, i] + j),
                            -(self.vars[X, OP.BOT] + j),
                        )
                    )
                    self.add_clause(
                        (
                            -(self.vars[X, OP.OR] + i),
                            -(self.vars[V, 2, i] + j),
                            -(self.vars[X, OP.BOT] + j + 1),
                        )
                    )

    def evaluation_constraints(self, tt: int):
        tree = all_trees(self.k)[tt]

        for a in range(self.inst.A.max_ind):
            for i in range(self.k):
                if OP.NEG in self.inst.op_b() and len(tree[i]) == 1:
                    self.add_clause(
                        (
                            -(self.vars[X, OP.NEG] + i),
                            -(self.vars[Z, a] + i),
                            -(self.vars[Z, a] + tree[i][0]),
                        )
                    )
                    self.add_clause(
                        (
                            -(self.vars[X, OP.NEG] + i),
                            (self.vars[Z, a] + i),
                            (self.vars[Z, a] + tree[i][0]),
                        )
                    )

                if OP.AND in self.inst.op_b() and len(tree[i]) == 2:
                    self.add_clause(
                        (
                            -(self.vars[X, OP.AND] + i),
                            -(self.vars[Z, a] + i),
                            self.vars[Z, a] + tree[i][0],
                        )
                    )
                    self.add_clause(
                        (
                            -(self.vars[X, OP.AND] + i),
                            -(self.vars[Z, a] + i),
                            self.vars[Z, a] + tree[i][0] + 1,
                        )
                    )
                    self.add_clause(
                        (
                            -(self.vars[X, OP.AND] + i),
                            (self.vars[Z, a] + i),
                            -(self.vars[Z, a] + tree[i][0] + 1),
                            -(self.vars[Z, a] + tree[i][0]),
                        )
                    )

                if OP.OR in self.inst.op_b() and len(tree[i]) == 2:
                    self.add_clause(
                        (
                            -(self.vars[X, OP.OR] + i),
                            (self.vars[Z, a] + i),
                            -(self.vars[Z, a] + tree[i][0]),
                        )
                    )
                    self.add_clause(
                        (
                            -(self.vars[X, OP.OR] + i),
                            (self.vars[Z, a] + i),
                            -(self.vars[Z, a] + tree[i][0] + 1),
                        )
                    )
                    self.add_clause(
                        (
                            -(self.vars[X, OP.OR] + i),
                            -(self.vars[Z, a] + i),
                            (self.vars[Z, a] + tree[i][0] + 1),
                            (self.vars[Z, a] + tree[i][0]),
                        )
                    )

                if OP.ALL in self.inst.op_r() and len(tree[i]) == 1:
                    for r in self.inst.sigma.rolenames:
                        successors = [b for (b, p) in self.inst.A.rn_ext[a] if p == r]
                        if len(successors) == 0:
                            # Optimization: most individuals don't have successors
                            self.add_clause(
                                (-(self.vars[X, OP.ALL, r] + i), (self.vars[Z, a] + i))
                            )
                        else:
                            self.add_clause(
                                [
                                    -(self.vars[X, OP.ALL, r] + i),
                                    (self.vars[Z, a] + i),
                                ]
                                + [-(self.vars[Z, b] + tree[i][0]) for b in successors]
                            )
                            for b in successors:
                                self.add_clause(
                                    (
                                        -(self.vars[X, OP.ALL, r] + i),
                                        -(self.vars[Z, a] + i),
                                        self.vars[Z, b] + tree[i][0],
                                    )
                                )

                if OP.EX in self.inst.op_r() and len(tree[i]) == 1:
                    for r in self.inst.sigma.rolenames:
                        successors = [b for (b, p) in self.inst.A.rn_ext[a] if p == r]
                        if len(successors) == 0:
                            # Optimization: most individuals don't have successors
                            self.add_clause(
                                (-(self.vars[X, OP.EX, r] + i), -(self.vars[Z, a] + i))
                            )
                        else:
                            self.add_clause(
                                [
                                    -(self.vars[X, OP.EX, r] + i),
                                    -(self.vars[Z, a] + i),
                                ]
                                + [(self.vars[Z, b] + tree[i][0]) for b in successors]
                            )
                            for b in successors:
                                self.add_clause(
                                    (
                                        -(self.vars[X, OP.EX, r] + i),
                                        (self.vars[Z, a] + i),
                                        -(self.vars[Z, b] + tree[i][0]),
                                    )
                                )

                if OP.LE in self.inst.op_q() and len(tree[i]) == 1:
                    for r in self.inst.sigma.rolenames:
                        successors = [b for (b, p) in self.inst.A.rn_ext[a] if p == r]
                        for q in range(1, self.max_q_per_r[r] + 2):
                            if len(successors) <= q:
                                # Optimization: most individuals don't have successors
                                self.add_clause(
                                    (
                                        -(self.vars[X, OP.LE, r, q] + i),
                                        (self.vars[Z, a] + i),
                                    )
                                )
                            else:
                                enc = CardEnc.atmost(
                                    [self.vars[Z, b] + tree[i][0] for b in successors],
                                    bound=q,
                                    top_id=self.max_var,
                                )
                                self.max_var = max(enc.nv, self.max_var)
                                for cl in enc.clauses:
                                    self.add_clause(
                                        [
                                            -(self.vars[X, OP.LE, r, q] + i),
                                            -(self.vars[Z, a] + i),
                                        ]
                                        + cl
                                    )
                                enc = CardEnc.atleast(
                                    [self.vars[Z, b] + tree[i][0] for b in successors],
                                    bound=q + 1,
                                    top_id=self.max_var,
                                )
                                self.max_var = max(enc.nv, self.max_var)
                                for cl in enc.clauses:
                                    self.add_clause(
                                        [
                                            -(self.vars[X, OP.LE, r, q] + i),
                                            (self.vars[Z, a] + i),
                                        ]
                                        + cl
                                    )

                if OP.GE in self.inst.op_q() and len(tree[i]) == 1:
                    for r in self.inst.sigma.rolenames:
                        successors = [b for (b, p) in self.inst.A.rn_ext[a] if p == r]
                        for q in range(2, self.max_q_per_r[r] + 2):
                            if len(successors) == 0 or len(successors) < q:
                                # Optimization: most individuals don't have successors
                                self.add_clause(
                                    (
                                        -(self.vars[X, OP.GE, r, q] + i),
                                        -(self.vars[Z, a] + i),
                                    )
                                )
                            else:
                                enc = CardEnc.atleast(
                                    [self.vars[Z, b] + tree[i][0] for b in successors],
                                    bound=q,
                                    top_id=self.max_var,
                                )
                                self.max_var = max(enc.nv, self.max_var)
                                for cl in enc.clauses:
                                    self.add_clause(
                                        [
                                            -(self.vars[X, OP.GE, r, q] + i),
                                            -(self.vars[Z, a] + i),
                                        ]
                                        + cl
                                    )
                                enc = CardEnc.atmost(
                                    [self.vars[Z, b] + tree[i][0] for b in successors],
                                    bound=q - 1,
                                    top_id=self.max_var,
                                )
                                self.max_var = max(enc.nv, self.max_var)
                                for cl in enc.clauses:
                                    self.add_clause(
                                        [
                                            -(self.vars[X, OP.GE, r, q] + i),
                                            (self.vars[Z, a] + i),
                                        ]
                                        + cl
                                    )

                if len(tree[i]) == 0:
                    self.add_clause(
                        (-(self.vars[X, OP.TOP] + i), (self.vars[Z, a] + i))
                    )
                    self.add_clause(
                        (-(self.vars[X, OP.BOT] + i), -(self.vars[Z, a] + i))
                    )

        for i in range(self.k):
            if len(tree[i]) != 0:
                continue
            for tp in self.types:
                for cn in self.inst.sigma.conceptnames:
                    if cn in tp:
                        self.add_clause((-(self.vars[X, cn] + i), self.vars[X, tp] + i))
                    if cn not in tp:
                        self.add_clause(
                            (-(self.vars[X, cn] + i), -(self.vars[X, tp] + i))
                        )

        for a in range(self.inst.A.max_ind):
            tp = frozenset(
                {
                    cn
                    for cn in self.inst.sigma.conceptnames
                    if a in self.inst.A.cn_ext[cn]
                }
            )
            assert tp in self.types
            for i in range(self.k):
                if len(tree[i]) != 0:
                    continue
                self.add_clause((-(self.vars[X, tp] + i), self.vars[Z, a] + i))
                self.add_clause(
                    (
                        (self.vars[X, tp] + i),
                        -(self.vars[Z, a] + i),
                    )
                )

    def fitting_constraints_approximate(self, n: int):
        assert self.solver
        lits = [self.vars[Z, a] for a in self.inst.P] + [
            -self.vars[Z, b] for b in self.inst.N
        ]

        enc = CardEnc.atleast(
            lits, bound=n, top_id=self.max_var, encoding=EncType.kmtotalizer
        )
        self.max_var = max(enc.nv, self.max_var)
        for clause in enc.clauses:
            self.solver.add_clause(clause)

    def model_n(self) -> int:
        assert self.solver and self.solver.get_status()
        # Return the number of positive/negative examples that is claimed to be covered by a model
        m = self.solver.get_model()
        assert isinstance(m, list)

        res: int = 0
        for p in self.inst.P:
            if self.vars[Z, p] + 0 in m:
                res += 1

        for n in self.inst.N:
            if self.vars[Z, n] + 0 not in m:
                res += 1
        return res

    def model_extension(self) -> frozenset[int]:
        assert self.solver and self.solver.get_status()
        # Return the number of positive/negative examples that is claimed to be covered by a model
        m = self.solver.get_model()
        assert isinstance(m, list)

        res: set[int] = set()
        for p in self.inst.P:
            if self.vars[Z, p] + 0 in m:
                res.add(p)

        for n in self.inst.N:
            if self.vars[Z, n] + 0 in m:
                res.add(n)
        return frozenset(res)

    def nodelabel(self, i: int) -> tuple[OP, int, str]:
        assert self.solver and self.solver.get_status()
        m = self.solver.get_model()
        assert isinstance(m, list)
        if (self.vars[X, OP.TOP] + i) in m:
            return (OP.TOP, 0, "")
        if (self.vars[X, OP.BOT] + i) in m:
            return (OP.BOT, 0, "")
        for cn in self.inst.sigma.conceptnames:
            if (self.vars[X, cn] + i) in m:
                return (OP.CN, 0, cn)
        for op in self.inst.op_b():
            if (self.vars[X, op] + i) in m:
                return (op, 0, "")
        if OP.EX in self.inst.op:
            for r in self.inst.sigma.rolenames:
                if (self.vars[X, OP.EX, r] + i) in m:
                    return (OP.EX, 0, r)
        if OP.ALL in self.inst.op:
            for r in self.inst.sigma.rolenames:
                if (self.vars[X, OP.ALL, r] + i) in m:
                    return (OP.ALL, 0, r)
        if OP.LE in self.inst.op:
            for r in self.inst.sigma.rolenames:
                for q in range(1, self.max_q_per_r[r] + 2):
                    if (self.vars[X, OP.LE, r, q] + i) in m:
                        return (OP.LE, q, r)
        if OP.GE in self.inst.op:
            for r in self.inst.sigma.rolenames:
                for q in range(2, self.max_q_per_r[r] + 2):
                    if (self.vars[X, OP.GE, r, q] + i) in m:
                        return (OP.GE, q, r)
        assert False

    def modelToTree(self, i: int = 0) -> ALCConcept:
        assert self.solver and self.solver.get_status()
        m = self.solver.get_model()
        assert isinstance(m, list)

        (op, q, r) = self.nodelabel(i)

        children: list[ALCConcept] = []
        for j in range(i + 1, self.k):
            if (self.vars[V, 1, i] + j) in m:
                children.append(self.modelToTree(j))
            if j < self.k - 1 and (self.vars[V, 2, i] + j) in m:
                children.append(self.modelToTree(j))
                children.append(self.modelToTree(j + 1))
        return ALCConcept(op, r, q, tuple(children))


ApproxTask = tuple[ALCSATEncoding, int, int, int]


def solve_approx(task: ApproxTask):
    enc, k, min_n, tt = task

    n = max(len(enc.inst.P), len(enc.inst.N), min_n)

    best_sol = None
    best_f1 = 0
    best_accuracy = 0
    best_n = 0
    enc.types = cn_types(enc.inst.A, enc.inst.sigma)
    enc.create_vars()
    enc.syn_tree_encoding(tt)
    enc.disable_topbot()
    enc.evaluation_constraints(tt)
    enc.symmetry_breaking()

    enc.solver = Solver(name="g4", incr=True, bootstrap_with=enc.clauses)

    while n <= len(enc.inst.P) + len(enc.inst.N):
        enc.fitting_constraints_approximate(n)

        if not enc.solver.solve():
            return best_accuracy, best_f1, best_n, k, best_sol

        best_sol = enc.modelToTree()
        extension = enc.model_extension()

        best_accuracy = enc.inst.accuracy(extension)
        best_f1 = enc.inst.f1score(extension)
        best_n = enc.model_n()
        n = best_n + 1

    return best_accuracy, best_f1, best_n, k, best_sol


class FittingALC:
    def __init__(
        self,
        A: Structure,
        max_k: int,
        P: list[int],
        N: list[int],
        op: Iterable[OP] = ALC_OP,        
        workers: int = 1,
        max_q: int = 2,
        bisim_reduction: bool = True,
        clustering: ThresholdMethod = ThresholdMethod.INTERVALS,
        max_thresholds: int = 10,
        exclude_atomic: Iterable[OP] = []
    ):
        self.max_k: int = max_k
        self.inst: Instance = Instance(
            A, P, N, non_empty_symbols(A), frozenset(op), max_q=max_q
        )
        self.workers: int = workers
        self.clustering = clustering
        self.bisim_reduction = bisim_reduction
        self.max_thresholds = max_thresholds
        self.exclude_atomic = exclude_atomic

    def solve(self):
        acc, _, _ = self.solve_incr(self.max_k, self.max_k)
        return acc == 1.0

    def solve_incr(self, max_k: int, start_k: int = 1, timeout: float = -1):
        return self.solve_incr_approx(
            max_k, start_k, len(self.inst.P) + len(self.inst.N), timeout=timeout
        )

    def solve_incr_approx(
        self, max_k: int, start_k: int = 1, min_n: int = 1, timeout: float = -1
    ) -> tuple[float, int, ALCConcept]:
        time_start = time.time()
        k: int = start_k
        n: int = max(len(self.inst.P), len(self.inst.N), min_n)
        best_sol: ALCConcept = ALCConcept(OP.TOP, "", 0, tuple())
        best_acc = 0
        dt = time.time() - time_start

        if OP.INV in self.inst.op:
            self.inst, reverse_inverse_mapping = encode_inverses(self.inst)
        else:
            reverse_inverse_mapping: dict[str, str] = {}

        self.inst.sigma = determine_relevant_symbols(
            self.inst.A, self.inst.P + self.inst.N, 1, max_k - 1
        )

        self.inst = restrict_neighborhood(self.inst, max_k)

        if OP.DGEQ in self.inst.op:
            self.inst, reverse_data_mapping = encode_dataproperties(
                self.inst, clustering=self.clustering, max_k=self.max_k, max_thresholds=self.max_thresholds
            )

        if self.bisim_reduction:
            self.inst = bisimulation_reduction(self.inst, max_k)

        self.inst = prune_conceptnames(self.inst)

        inters = set(self.inst.P).intersection(set(self.inst.N))
        if len(inters) > 0:
            print(f"=== {len(inters)} individuals both positive and negative")

        with ProcessPoolExecutor(self.workers) as p:
            while k <= max_k and (dt < timeout or timeout == -1) and best_acc < 1.0:
                enc = ALCSATEncoding(self.inst)
                enc.exclude_atomic = self.exclude_atomic
                enc.k = k

                tasks: list[ApproxTask] = [
                    (enc, k, n, tt) for tt in range(len(all_trees(k)))
                ]

                fts = [p.submit(solve_approx, task) for task in tasks]

                dt = time.time() - time_start
                if timeout != -1:
                    remaining_time = timeout - dt
                else:
                    remaining_time = None

                progress = 0
                print(f"Searching with k = {k}, progress {progress}/{len(tasks)}")
                try:
                    for ft in concurrent.futures.as_completed(
                        fts, timeout=remaining_time
                    ):
                        k_acc, k_f1, k_n, _, k_sol = ft.result()
                        progress += 1

                        print(
                            f"Searching with k = {k}, progress {progress}/{len(tasks)}"
                        )

                        if k_acc > best_acc:
                            assert k_sol
                            best_sol = k_sol
                            best_acc = k_acc
                            print(
                                f"Satisfiable for k={k}, n={k_n}, acc={k_acc:.6f}, f1={k_f1:.6f}"
                            )
                            print(best_sol.to_tree())
                            n = k_n + 1
                except TimeoutError:
                    pass

                k += 1
                dt = time.time() - time_start

            # Really kill the SAT solver processes
            for proc in p._processes.values():
                proc.terminate()
            p.shutdown(wait=False, cancel_futures=True)

        decoded_sol = best_sol

        if OP.DGEQ in self.inst.op:
            decoded_sol = decode_dataproperties(best_sol, reverse_data_mapping)

        if OP.INV in self.inst.op:
            decoded_sol = decode_inverses(decoded_sol, reverse_inverse_mapping)

        return best_acc, k, decoded_sol


def perfect_fitting(
    inst: Instance,
    clustering: ThresholdMethod = ThresholdMethod.ALL_THRESHOLDS,
    max_thresholds: int = 10,
) -> tuple[float, int, ALCConcept]:
    orig_inst = inst
    inst, back = encode_dataproperties(inst, clustering, max_thresholds)

    sig = inst.sigma
    A = inst.A
    P = inst.P
    N = inst.N

    colors_alcq, cr = color_refinement(A, sig, True, -1)

    pos_colors = {}
    neg_colors = {}
    for p in P:
        c = colors_alcq[p]
        if c not in pos_colors:
            pos_colors[c] = 0
        pos_colors[c] += 1

    for n in N:
        c = colors_alcq[n]
        if c not in neg_colors:
            neg_colors[c] = 0
        neg_colors[c] += 1

    disj = list()
    for cp in pos_colors.keys():
        if cp in neg_colors and pos_colors[cp] < neg_colors[cp]:
            # Including this positive example would include a lot of negative
            # examples and thus not be beneficial for accuracy
            continue

        conj = list()
        for cn in neg_colors.keys():
            if cp == cn:
                continue

            res = extract_concept(cr, cp, cn, A)
            for p in P:
                if colors_alcq[p] == cp:
                    assert res.mc(inst.A, p)
            conj.append(res)

        conj = simplify_conj(conj, A)
        d = merge_conj(deduplicate(conj), OP.AND)
        disj.append(d)

    c = merge_conj(deduplicate(disj), OP.OR)
    c = decode_dataproperties(c, back)

    ext: set[int] = set()
    for a in P + N:
        if c.mc(orig_inst.A, a):
            ext.add(a)

    acc = orig_inst.accuracy(frozenset[int](ext))

    return acc, c.size(), c
