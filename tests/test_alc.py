from alcsat.fitting_alc import FittingALC, all_trees
from alcsat.instance import ALCQ_OP, OP, ALCConcept, Instance
from alcsat.preprocessing import (
    bisimulation_reduction,
    color_refinement,
    extract_concept,
)
from alcsat.structures import Signature, Structure


def test_trees():
    # See https://en.wikipedia.org/wiki/Wedderburn%E2%80%93Etherington_number
    assert len(all_trees(10)) == 207


def test_color_refinement():
    rn2 = {i: set() for i in range(7)}
    rn2[0] = {(1, "r")}
    rn2[2] = {(3, "r"), (4, "r")}
    rn2[5] = {(6, "r")}
    cn2 = {"A": {6}}
    A2 = Structure(7, cn2, rn2, {i: list() for i in range(6)}, {}, {})

    colors_alcq, _ = color_refinement(A2, Signature(["A"], ["r"]), True, 10)

    assert colors_alcq[0] != colors_alcq[2]
    assert colors_alcq[0] != colors_alcq[5]
    assert colors_alcq[2] != colors_alcq[5]
    assert colors_alcq[1] == colors_alcq[3]

    colors_alc, _ = color_refinement(A2, Signature(["A"], ["r"]), False, 10)
    assert colors_alc[0] == colors_alc[2]
    assert colors_alc[0] != colors_alc[5]
    assert colors_alc[1] == colors_alc[3]


def test_alcq_filtration():
    rn2 = {i: set() for i in range(7)}
    rn2[0] = {(1, "r"), (2, "r")}
    A2 = Structure(3, {}, rn2, {i: list() for i in range(6)}, {}, {})

    inst = Instance(A2, [0], [0], Signature([], ["r"]), ALCQ_OP, max_q=4)
    inst = bisimulation_reduction(inst, 10)

    c = ALCConcept(OP.GE, "r", 2, (ALCConcept(OP.TOP, "", 0, tuple()),))

    assert c.mc(inst.A, inst.P[0])


def test_alcq_filtration2():
    rn2 = {i: set() for i in range(7)}
    rn2[0] = {(2, "r")}
    rn2[1] = {(3, "r"), (4, "r")}
    rn2[2] = {(5, "r")}
    rn2[3] = {(5, "r")}
    rn2[4] = {(5, "r")}
    A2 = Structure(6, {}, rn2, {i: list() for i in range(6)}, {}, {})

    inst = Instance(A2, [1], [0], Signature([], ["r"]), ALCQ_OP, max_q=2)
    inst = bisimulation_reduction(inst, 10)

    c = ALCConcept(OP.GE, "r", 2, (ALCConcept(OP.TOP, "", 0, tuple()),))

    assert c.mc(inst.A, inst.P[0])
    assert not c.mc(inst.A, inst.N[0])


def test1():
    A1 = Structure(
        3,
        {"A": {0, 1}, "B": {0, 2}},
        {i: set() for i in range(3)},
        {i: list() for i in range(6)},
        {},
        {},
    )
    P1 = [0]
    N1 = [1, 2]
    i = (A1, 3, P1, N1)
    f = FittingALC(*i, op={OP.EX, OP.ALL, OP.OR, OP.AND})
    assert f.solve()


def test2():
    rn2 = {i: set() for i in range(3)}
    rn2[0] = {(2, "r")}
    cn2 = {"A": {0, 1, 2}, "B": {0, 1}}
    A2 = Structure(3, cn2, rn2, {i: list() for i in range(6)}, {}, {})
    P2 = [0]
    N2 = [1]
    i = (A2, 3, P2, N2)
    f = FittingALC(*i, op={OP.EX, OP.ALL, OP.OR, OP.AND})
    assert f.solve()


def test3():
    A3 = Structure(
        3,
        {"A": {1}, "B": {2}},
        {i: set() for i in range(3)},
        {i: list() for i in range(6)},
        {},
        {},
    )
    P3 = [1, 2]
    N3 = [0]
    i = (A3, 3, P3, N3)
    f = FittingALC(*i, op={OP.EX, OP.ALL, OP.OR, OP.AND})
    assert f.solve()


def test4():
    rn4 = dict()
    rn4[0] = {(0, "r")}
    rn4[1] = {(2, "r")}
    rn4[2] = {(3, "r")}
    rn4[3] = {}
    A4 = Structure(4, {"A": {0}, "B": {3}}, rn4, {i: list() for i in range(6)}, {}, {})

    P4 = [1]
    N4 = [0]
    i = (A4, 3, P4, N4)
    f = FittingALC(*i, op={OP.EX, OP.ALL, OP.OR, OP.AND})
    assert f.solve()


def test5():
    A5 = Structure(
        2,
        {"A": {1}, "B": {1}},
        {i: set() for i in range(2)},
        {i: list() for i in range(6)},
        {},
        {},
    )
    P5 = [0]
    N5 = []
    i = (A5, 1, P5, N5)
    f = FittingALC(*i, op={OP.EX, OP.ALL, OP.OR, OP.AND})
    assert f.solve()


def test6():
    A6 = Structure(
        3,
        {"A": {1}, "B": {0, 1}},
        {i: set() for i in range(3)},
        {i: list() for i in range(6)},
        {},
        {},
    )
    P6 = [0]
    N6 = [1]
    i = (A6, 2, P6, N6)
    f = FittingALC(*i, op={OP.EX, OP.ALL, OP.OR, OP.AND, OP.NEG})
    assert f.solve()


def test_alcq():
    d = {
        1: {(2, "r"), (2, "r")},
        4: {(5, "r"), (6, "r")},
        7: {(8, "r"), (9, "r"), (10, "r"), (11, "r")},
    }
    for i in [0, 2, 3, 5, 6, 8, 9, 10, 11]:
        d[i] = {}
    A = Structure(
        11,
        {"A": {2, 8}, "B": {3, 9}, "C": {5, 10}, "D": {6, 11}},
        d,
        {i: list() for i in range(12)},
        {},
        {},
    )
    P = [1, 4]
    N = [7]
    i = (A, 2, P, N)
    f = FittingALC(*i, op={OP.EX, OP.ALL, OP.OR, OP.AND, OP.NEG, OP.LE, OP.GE})
    assert f.solve()


def testEx():
    A = Structure(
        5,
        {"A": {2, 4, 5}, "B": {3}},
        {
            0: {(2, "r"), (3, "r")},
            1: {(4, "r"), (5, "r")},
            2: set(),
            3: set(),
            4: set(),
            5: set(),
        },
        {i: list() for i in range(6)},
        {},
        {},
    )

    i = (A, 2, [0], [1])
    f = FittingALC(*i, op={OP.EX})
    assert f.solve()

    i2 = (A, 2, [1], [0])
    f2 = FittingALC(*i2, op={OP.EX})
    assert not f2.solve()


def testAnd():
    A = Structure(
        5,
        {"A": {1, 2, 3}, "B": {1, 3, 4}, "C": {1, 2, 4}},
        {0: set(), 1: set(), 2: set(), 3: set(), 4: set()},
        {i: list() for i in range(6)},
        {},
        {},
    )

    i = (A, 4, [1], [2, 3, 4])
    f = FittingALC(*i, op={OP.AND})
    assert not f.solve()

    i2 = (A, 5, [1], [2, 3, 4])
    f2 = FittingALC(*i2, op={OP.AND})
    assert f2.solve()


def testAll():
    A = Structure(
        5,
        {"A": {2, 4, 5}, "B": {3}},
        {
            0: {(2, "r"), (3, "r")},
            1: {(4, "r"), (5, "r")},
            2: set(),
            3: set(),
            4: set(),
            5: set(),
        },
        {i: list() for i in range(6)},
        {},
        {},
    )

    i = (A, 2, [0], [1])
    f = FittingALC(*i, op={OP.ALL})
    assert not f.solve()

    i2 = (A, 2, [1], [0])
    f2 = FittingALC(*i2, op={OP.ALL})
    assert f2.solve()


def testInv1():
    A = Structure(
        3,
        {},
        {0: set(), 1: set(), 2: {(0, "r")}},
        {i: list() for i in range(6)},
        {},
        {},
    )

    i = (A, 2, [0], [1])
    f = FittingALC(*i, op={OP.EX, OP.INV, OP.NEG})
    assert f.solve()

    i = (A, 2, [1], [0])
    f = FittingALC(*i, op={OP.ALL, OP.INV})
    assert f.solve()


def testInv2():
    A = Structure(
        8,
        {},
        {
            0: set(),
            1: {(4, "r")},
            2: {(4, "r")},
            3: {(4, "r")},
            4: set(),
            5: {(7, "r")},
            6: {(7, "r")},
            7: set(),
        },
        {i: list() for i in range(8)},
        {},
        {},
    )

    i = (A, 3, [1], [5])
    f = FittingALC(*i, op={OP.EX, OP.GE, OP.INV}, max_q=3)
    assert f.solve()

    i = (A, 3, [5], [1])
    f = FittingALC(*i, op={OP.EX, OP.LE, OP.INV}, max_q=3)
    assert f.solve()


def testSize():
    k = 10
    # TODO: the SAT formula for this takes a surprising amount of time to solve
    # i.e. it is not instant
    # I believe if we modify our encoding such that this becomes instant, we can gain
    # a lot of speed on realistic benchmarks
    A = Structure(
        max_ind=k,
        cn_ext={},
        rn_ext={i: {(i + 1, "r")} for i in range(k - 1)},
        indmap={},
        nsmap={},
        dp_ext={i: list() for i in range(k + 1)},
    )
    A.rn_ext[k - 1] = set()

    i = (A, k, [0], [1])
    f = FittingALC(*i, op={OP.EX, OP.AND})
    assert f.solve()


def testExtraction():
    rn2: dict[int, set[tuple[int, str]]] = {i: set() for i in range(7)}
    rn2[0] = {(1, "r")}
    rn2[2] = {(3, "r"), (4, "r")}
    rn2[5] = {(6, "r")}
    cn2 = {"A": {6}}
    A2 = Structure(7, cn2, rn2, {i: [] for i in range(6)}, {}, {})

    colors_alcq, cr = color_refinement(A2, Signature(["A"], ["r"]), True, -1)

    for a in range(A2.max_ind):
        for b in range(A2.max_ind):
            if colors_alcq[a] != colors_alcq[b]:
                res = extract_concept(cr, colors_alcq[a], colors_alcq[b], A2)

                assert res.mc(A2, a)
                assert not res.mc(A2, b)



if __name__ == "__main__":
    testExtraction()
