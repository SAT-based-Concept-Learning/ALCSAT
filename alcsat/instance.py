from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from alcsat.structures import Signature, Structure, ind


class OP(IntEnum):
    CN = -1
    TOP = 0
    BOT = 1
    NEG = 2
    AND = 3
    OR = 4
    EX = 5
    ALL = 6
    LE = 7
    GE = 8
    DGEQ = 9
    INV = 10


ALC_OP = frozenset({OP.NEG, OP.AND, OP.OR, OP.EX, OP.ALL})
ALC_OP_B = frozenset({OP.NEG, OP.AND, OP.OR})
ALC_OP_R = frozenset({OP.EX, OP.ALL})
ALC_OP_Q = frozenset({OP.LE, OP.GE})
ALCQ_OP = frozenset({OP.NEG, OP.AND, OP.OR, OP.EX, OP.ALL, OP.LE, OP.GE})

d_op = {
    0: "TOP",
    1: "BOT",
    2: "NEG",
    3: "AND",
    4: "OR",
    5: "EX",
    6: "ALL",
    7: "LE",
    8: "GE",
    9: "DGEQ",
    10: "INV",
}


@dataclass(slots=True, frozen=True)
class ALCConcept:
    operation: OP
    name: str
    value: Any
    children: tuple["ALCConcept", ...]
    inverse: bool = False

    def to_tree_int(self) -> list[str]:
        if self.operation == OP.CN:
            # concept name
            res = [self.name]
        elif self.operation in {OP.ALL, OP.EX}:
            if self.inverse:
                res = [f"{d_op[self.operation]}.inv({self.name})"]
            else:
                res = [f"{d_op[self.operation]}.{self.name}"]
        elif self.operation in {OP.GE, OP.LE}:
            if self.inverse:
                res = [f"{d_op[self.operation]}{self.value} inv({self.name})"]
            else:
                res = [f"{d_op[self.operation]}{self.value} {self.name}"]
        elif self.operation in {OP.DGEQ}:
            res = [f"({self.name} >= {self.value})"]
        else:
            res = [f"{d_op[self.operation]}"]

        for c in self.children:
            cs = c.to_tree_int()
            res.append(" +-- " + cs[0])
            res.extend(["    " + s for s in cs[1:]])
        return res

    def to_tree(self) -> str:
        return "\n".join(self.to_tree_int())

    def to_dl_concept(self) -> str:
        if self.operation == OP.CN:
            return self.name
        if self.operation in {OP.ALL, OP.EX} and self.inverse:
            return f"{d_op[self.operation]}.inv({self.name}) {self.children[0].to_dl_concept()}"
        if self.operation in {OP.ALL, OP.EX} and not self.inverse:
            return (
                f"{d_op[self.operation]}.{self.name} {self.children[0].to_dl_concept()}"
            )
        if self.operation in {OP.GE, OP.LE} and self.inverse:
            return f"{d_op[self.operation]}{self.value}.inv({self.name}) {self.children[0].to_dl_concept()}"
        if self.operation in {OP.GE, OP.LE} and not self.inverse:
            return f"{d_op[self.operation]}{self.value}.{self.name} {self.children[0].to_dl_concept()}"
        if self.operation in {OP.DGEQ}:
            return f"({self.name} >= {self.value})"
        if self.operation in {OP.NEG}:
            return f"NEG {self.children[0].to_dl_concept()}"
        if self.operation in {OP.AND, OP.OR}:
            return f"({self.children[0].to_dl_concept()} {d_op[self.operation]} {self.children[1].to_dl_concept()})"
        return d_op[self.operation]

    def evo_size(self) -> int:
        if self.operation in {OP.ALL, OP.EX, OP.GE, OP.LE, OP.DGEQ}:
            return 2 + sum(c.evo_size() for c in self.children)
        else:
            return 1 + sum(c.evo_size() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def mc(self, A: Structure, a: int) -> bool:
        assert a in ind(A)

        if self.operation == OP.TOP:
            return True
        if self.operation == OP.BOT:
            return False
        if self.operation == OP.CN:
            return a in A.cn_ext[self.name]
        if self.operation == OP.AND:
            assert len(self.children) == 2
            return self.children[0].mc(A, a) and self.children[1].mc(A, a)
        if self.operation == OP.OR:
            assert len(self.children) == 2
            return self.children[0].mc(A, a) or self.children[1].mc(A, a)
        if self.operation == OP.NEG:
            assert len(self.children) == 1
            return not self.children[0].mc(A, a)
        if self.operation == OP.EX and not self.inverse:
            assert len(self.children) == 1
            cnt = len(
                [
                    b
                    for (b, r) in A.rn_ext[a]
                    if r == self.name and self.children[0].mc(A, b)
                ]
            )
            return cnt >= 1
        if self.operation == OP.EX and self.inverse:
            assert len(self.children) == 1
            cnt = len(
                [
                    b
                    for b in ind(A)  # TODO: Inefficient
                    if (a, self.name) in A.rn_ext[a] and self.children[0].mc(A, b)
                ]
            )
            return cnt >= 1
        if self.operation == OP.ALL and not self.inverse:
            assert len(self.children) == 1
            cnt = len(
                [
                    b
                    for (b, r) in A.rn_ext[a]
                    if r == self.name and not self.children[0].mc(A, b)
                ]
            )
            return cnt == 0
        if self.operation == OP.ALL and self.inverse:
            assert len(self.children) == 1
            cnt = len(
                [
                    b
                    for b in ind(A)  # TODO: Inefficient
                    if (a, self.name) in A.rn_ext[a] and not self.children[0].mc(A, b)
                ]
            )
            return cnt == 0
        if self.operation == OP.GE and not self.inverse:
            assert len(self.children) == 1
            cnt = len(
                [
                    b
                    for (b, r) in A.rn_ext[a]
                    if r == self.name and self.children[0].mc(A, b)
                ]
            )
            return cnt >= self.value
        if self.operation == OP.GE and self.inverse:
            assert len(self.children) == 1
            cnt = len(
                [
                    b
                    for b in ind(A)  # TODO: Inefficient
                    if (a, self.name) in A.rn_ext[a] and self.children[0].mc(A, b)
                ]
            )
            return cnt >= self.value
        if self.operation == OP.LE and not self.inverse:
            assert len(self.children) == 1
            cnt = len(
                [
                    b
                    for (b, r) in A.rn_ext[a]
                    if r == self.name and self.children[0].mc(A, b)
                ]
            )
            return cnt <= self.value
        if self.operation == OP.LE and self.inverse:
            assert len(self.children) == 1
            cnt = len(
                [
                    b
                    for b in ind(A)  # TODO: Inefficient
                    if (a, self.name) in A.rn_ext[a] and self.children[0].mc(A, b)
                ]
            )
            return cnt <= self.value
        if self.operation == OP.DGEQ:
            assert len(self.children) == 0

            val: None | Any = None
            for v, _, r in A.dp_ext[a]:
                if r == self.name:
                    val = v
            if val is None:
                return False
            return val >= self.value
        assert False


@dataclass(slots=True)
class Instance:
    A: Structure
    P: list[int]
    N: list[int]
    sigma: Signature
    op: frozenset[OP]
    max_q: int

    def op_b(self):
        return self.op.intersection(ALC_OP_B)

    def op_r(self):
        return self.op.intersection(ALC_OP_R)

    def op_q(self):
        return self.op.intersection(ALC_OP_Q)

    def accuracy(self, st: frozenset[int]) -> float:
        tp = 0
        tn = 0

        for a in self.P:
            if a in st:
                tp += 1

        for a in self.N:
            if a not in st:
                tn += 1

        return (tp + tn) / (len(self.P) + len(self.N))

    def f1score(self, st: frozenset[int]) -> float:
        tp = 0
        fn = 0
        fp = 0

        for a in self.P:
            if a in st:
                tp += 1
            else:
                fn += 1

        for a in self.N:
            if a in st:
                fp += 1

        return (2 * tp) / (2 * tp + fp + fn)
