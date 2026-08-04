from enum import Enum
from collections import defaultdict
from typing import Any
from alcsat.instance import ALCConcept, Instance, OP
from alcsat.structures import Signature, Structure, ind
from scipy.cluster.vq import kmeans
from numpy import array


class ThresholdMethod(Enum):
    KMEANS = 0
    INTERVALS = -1
    NEIGHBOORHOOD_KMEANS = 1
    ALL_THRESHOLDS = 2
    NONE = 3


def prune_conceptnames(inst: Instance) -> Instance:
    A = inst.A
    sigma = Signature(inst.sigma.conceptnames, inst.sigma.rolenames)

    redundant: set[str] = set()
    for cn1 in sigma.conceptnames:
        for cn2 in sigma.conceptnames:
            if cn1 < cn2 and A.cn_ext[cn1] == A.cn_ext[cn2]:
                redundant.add(cn2)

    print(f"== Pruning {len(redundant)} redundant concept names")

    sigma.conceptnames = [cn for cn in sigma.conceptnames if cn not in redundant]

    return Instance(A, inst.P, inst.N, sigma, inst.op, inst.max_q)


def neighborhoods(inst: Instance, max_k: int):
    N_p: dict[int, set[int]] = defaultdict(set)
    N_n: dict[int, set[int]] = defaultdict(set)
    N_p[0] = set(inst.P)
    N_n[0] = set(inst.N)
    for i in range(1, max_k):
        for a in N_p[i - 1]:
            for b, r in inst.A.rn_ext[a]:
                N_p[i].add(b)
        for a in N_n[i - 1]:
            for b, r in inst.A.rn_ext[a]:
                N_n[i].add(b)
    return N_p, N_n


def cluster_neighborhoods(
    inst: Instance, neighborhoods: dict[int, set[int]], n_clusters
):
    values: dict[str, list[set[float]]] = defaultdict(list)
    result: dict[str, list[Any]] = defaultdict(list)
    for i, ab in neighborhoods.items():
        values2: dict[str, set[Any]] = defaultdict(set)
        for a in ab:
            for v, _, pp in inst.A.dp_ext[a]:
                values2[pp].add(v)
        for p, vs in values2.items():
            values[p].append(vs)
    for p, vss in values.items():
        for vs in vss:
            if len(vs) > n_clusters:
                centroids = sorted(kmeans(array(list(vs)), n_clusters)[0])
                for i in range(len(centroids) - 1):
                    result[p].append((centroids[i] + centroids[i + 1]) / 2)
    return result


def pick_data_intervals(
    ranges: dict[str, set[Any]], max_thresholds: int
) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}

    for p, values in ranges.items():
        if values == {True, False}:
            result[p] = [True]
        elif len(values) <= max_thresholds:
            result[p] = list(values)
        elif max_thresholds == 0:
            result[p] = []
        else:
            thresholds: list[Any] = list()
            vs = list(values)
            vs.sort()

            step = len(vs) // max_thresholds

            for i in range(max_thresholds):
                thresholds.append(vs[step * i])
            result[p] = thresholds
    return result


def pick_all_thresholds(
    ranges: dict[str, set[Any]], max_thresholds: int
) -> dict[str, set[Any]]:
    result: dict[str, set[Any]] = {}

    for p, values in ranges.items():
        result[p] = values
    return result


def pick_data_clusters(
    ranges: dict[str, set[Any]], max_thresholds: int
) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}

    for p, values in ranges.items():
        if values == {True, False}:
            result[p] = [True]
        elif len(values) <= 10:
            result[p] = list(values)
        else:
            vs = list(values)
            if not isinstance(vs[0], int) and not isinstance(vs[0], float):
                result[p] = list()
                continue

            # kmeans cannot process ints, convert to floats
            vs = [float(v) for v in vs]

            # TODO: does this handle dates? and strings
            centroids = sorted(kmeans(array(vs), max_thresholds)[0])
            result[p] = list()
            result[p].append(min(vs))
            for i in range(len(centroids) - 1):
                result[p].append((centroids[i] + centroids[i + 1]) / 2)
    return result


def decode_dataproperties(
    c: ALCConcept, reverse_mapping: dict[str, ALCConcept]
) -> ALCConcept:
    if c.operation == OP.CN and c.name in reverse_mapping:
        return reverse_mapping[c.name]

    nchildren = [decode_dataproperties(d, reverse_mapping) for d in c.children]

    return ALCConcept(c.operation, c.name, c.value, tuple(nchildren))


def encode_dataproperties(
    inst: Instance, clustering: ThresholdMethod, max_thresholds=10, max_k=10
) -> tuple[Instance, dict[str, ALCConcept]]:
    A = inst.A
    sigma = Signature(list(inst.sigma.conceptnames), list(inst.sigma.rolenames))

    ranges: dict[str, set[Any]] = {}

    for a in range(A.max_ind):
        for v, _, p in A.dp_ext[a]:
            if p not in ranges:
                ranges[p] = set()
            ranges[p].add(v)

    if clustering == ThresholdMethod.NONE:
        return inst, {}
    elif clustering == ThresholdMethod.INTERVALS:
        thresholds = pick_data_intervals(ranges, max_thresholds)
    elif clustering == ThresholdMethod.ALL_THRESHOLDS:
        thresholds = pick_all_thresholds(ranges, max_thresholds)
    elif clustering == ThresholdMethod.KMEANS:
        thresholds = pick_data_clusters(ranges, max_thresholds)
    elif clustering == ThresholdMethod.NEIGHBOORHOOD_KMEANS:
        N_p, N_n = neighborhoods(inst, max_k=max_k)
        result1 = cluster_neighborhoods(inst, N_p, (max_thresholds) // 2 + 1)
        result2 = cluster_neighborhoods(inst, N_n, max_thresholds // 2 + 1)
        for p, v in result2.items():
            if p in result1:
                result1[p].union(v)
            else:
                result1[p] = v
        thresholds = result1
    else:
        raise ValueError("A very specific bad thing happened.")

    # print(f"=== I choose thresholds: {thresholds}")

    B = Structure(A.max_ind, {}, {}, {}, A.indmap, A.nsmap)

    for cn in sigma.conceptnames:
        B.cn_ext[cn] = set(A.cn_ext[cn])

    for a in range(A.max_ind):
        B.rn_ext[a] = set(A.rn_ext[a])
        B.dp_ext[a] = []

    reverse_mapping: dict[str, ALCConcept] = {}

    for a in range(A.max_ind):
        for v, _, p in A.dp_ext[a]:
            for r in thresholds[p]:
                cn = f"{p}>={r}"
                if cn not in B.cn_ext:
                    B.cn_ext[cn] = set()
                    sigma.conceptnames.append(cn)
                    reverse_mapping[cn] = ALCConcept(OP.DGEQ, p, r, tuple())
                if v >= r:
                    B.cn_ext[cn].add(a)

    return Instance(B, inst.P, inst.N, sigma, inst.op, inst.max_q), reverse_mapping


def encode_inverses(inst: Instance) -> tuple[Instance, dict[str, str]]:
    assert OP.INV in inst.op

    from_inverse: dict[str, str] = {}
    to_inverse: dict[str, str] = {}

    A = inst.A
    sigma = Signature(list(inst.sigma.conceptnames), list(inst.sigma.rolenames))

    rns = list(sigma.rolenames)
    for rn in rns:
        inv_rn = f"inv({rn})"
        sigma.rolenames.append(inv_rn)
        from_inverse[inv_rn] = rn
        to_inverse[rn] = inv_rn

    B = Structure(A.max_ind, {}, {}, {}, {}, A.nsmap)

    for cn in sigma.conceptnames:
        B.cn_ext[cn] = set(A.cn_ext[cn])

    for a in range(A.max_ind):
        B.rn_ext[a] = set(A.rn_ext[a])
        B.dp_ext[a] = list(A.dp_ext[a])

    for a in range(A.max_ind):
        for b, r in A.rn_ext[a]:
            if r in sigma.rolenames:
                B.rn_ext[b].add((a, to_inverse[r]))

    return Instance(
        B, inst.P, inst.N, sigma, inst.op.difference({OP.INV}), inst.max_q
    ), from_inverse


def decode_inverses(c: ALCConcept, from_inverse: dict[str, str]) -> ALCConcept:
    if c.operation in {OP.ALL, OP.EX, OP.LE, OP.GE} and c.name in from_inverse:
        return ALCConcept(
            c.operation,
            from_inverse[c.name],
            c.value,
            tuple([decode_inverses(child, from_inverse) for child in c.children]),
            inverse=True,
        )

    return ALCConcept(
        c.operation,
        c.name,
        c.value,
        tuple([decode_inverses(child, from_inverse) for child in c.children]),
    )


def compute_multiplicities(
    color_register: dict[tuple[tuple[int, str], ...], int],
) -> tuple[dict[int, int], dict[tuple[int, int, str], int]]:
    color_multiplicities: dict[int, int] = defaultdict(lambda: 1)
    edge_multiplicities: dict[tuple[int, int, str], int] = defaultdict(int)

    for desc, c in color_register.items():
        for d, r in desc:
            if d != -1:
                edge_multiplicities[(c, d, r)] += 1

    for c, d, r in edge_multiplicities:
        color_multiplicities[d] = max(
            color_multiplicities[d], edge_multiplicities[(c, d, r)]
        )

    return color_multiplicities, edge_multiplicities


def color_refinement(
    A: Structure, sigma: Signature, alc_q: bool, iterations: int
) -> tuple[dict[int, int], list[dict[tuple[tuple[int, str], ...], int]]]:
    color: dict[int, int] = dict.fromkeys(range(A.max_ind), 0)
    color_register: list[dict[tuple[tuple[int, str], ...], int]] = []

    local_types: dict[int, list[tuple[int, str]]] = {a: [] for a in range(A.max_ind)}
    for cn in sigma.conceptnames:
        for a in A.cn_ext[cn]:
            local_types[a].append((-1, cn))

    i = 0
    while i <= iterations or iterations == -1:
        i += 1
        ncolor: dict[int, int] = {}
        color_register.append({})

        for a in range(A.max_ind):
            tp2 = list(local_types[a])
            for b, r in A.rn_ext[a]:
                tp2.append((color[b], r))
            if not alc_q:
                tp2 = list(set(tp2))
            tp2.sort()
            tpf2 = tuple(tp2)

            if tpf2 not in color_register[-1]:
                color_register[-1][tpf2] = len(color_register[-1])
            ncolor[a] = color_register[-1][tpf2]

        if color == ncolor:
            # No change happened
            return color, color_register

        color = ncolor

    return color, color_register


def bisimulation_reduction(inst: Instance, max_k: int) -> Instance:
    use_q = OP.LE in inst.op or OP.GE in inst.op

    color, color_register = color_refinement(inst.A, inst.sigma, use_q, max_k)

    final_colors = color_register[-1]

    color_multiplicities, edge_multiplicities = compute_multiplicities(final_colors)

    A = inst.A
    sigma = inst.sigma

    color2ind: dict[tuple[int, int], int] = {}
    for a in range(A.max_ind):
        c = color[a]
        for i in range(color_multiplicities[c]):
            if (c, i) not in color2ind:
                color2ind[(c, i)] = len(color2ind)

    B = Structure(len(color2ind), {}, {}, {}, {}, A.nsmap)

    for cn in sigma.conceptnames:
        B.cn_ext[cn] = set()
        for a in A.cn_ext[cn]:
            for i in range(color_multiplicities[color[a]]):
                B.cn_ext[cn].add(color2ind[(color[a], i)])

    for c in color.values():
        for i in range(color_multiplicities[c]):
            ca = color2ind[(c, i)]
            if ca not in B.rn_ext:
                B.rn_ext[ca] = set()
                B.dp_ext[ca] = []

    for (c, d, r), n in edge_multiplicities.items():
        for i in range(color_multiplicities[c]):
            ca = color2ind[(c, i)]
            for j in range(n):
                cb = color2ind[(d, j)]
                B.rn_ext[ca].add((cb, r))

    print(
        f"== Bisimulation reduction reduced from size {A.max_ind} to size {B.max_ind}"
    )

    return Instance(
        B,
        [color2ind[(color[p], 0)] for p in inst.P],
        [color2ind[(color[n], 0)] for n in inst.N],
        sigma,
        inst.op,
        inst.max_q,
    )


# Returns A restricted to individuals that can be reached in k steps from a
# Renames individuals
def restrict_to_neighborhood(
    k: int, A: Structure, starts: list[int]
) -> tuple[Structure, dict[int, int]]:
    cns = [cn for cn in A.cn_ext if A.cn_ext[cn]]

    # This has its own distance calculation to avoid computing the distance
    # for the entirety of A
    inds = set(starts)
    dist = dict.fromkeys(starts, 0)
    for r in range(k):
        step: set[int] = set()
        for i1 in inds:
            for i2, rn in A.rn_ext[i1]:
                step.add(i2)
        inds = inds.union(step)
        for i in step:
            if i in dist:
                dist[i] = min(r + 1, dist[i])
            else:
                dist[i] = r + 1

    mapping = {old_ind: new_ind for (new_ind, old_ind) in enumerate(inds)}

    n_indmap = {
        name: mapping[old_ind]
        for name, old_ind in A.indmap.items()
        if old_ind in mapping
    }

    B = Structure(
        max_ind=len(inds),
        cn_ext={cn: set() for cn in cns},
        rn_ext={a: set() for a in range(len(inds))},
        dp_ext={a: [] for a in range(len(inds))},
        indmap=n_indmap,
        nsmap=A.nsmap,
    )

    for cn in cns:
        B.cn_ext[cn] = {mapping[ind] for ind in A.cn_ext[cn] & inds}

    for i1 in inds:
        B.rn_ext[mapping[i1]] = set()
        B.dp_ext[mapping[i1]] = A.dp_ext[i1]
        for i2, rn in A.rn_ext[i1]:
            if i2 in inds and dist[i1] < k:
                B.rn_ext[mapping[i1]].add((mapping[i2], rn))

    return (B, mapping)


def restrict_neighborhood(inst: Instance, k: int) -> Instance:
    A, mapping = restrict_to_neighborhood(k - 1, inst.A, inst.P + inst.N)
    P2 = [mapping[a] for a in inst.P]
    N2 = [mapping[a] for a in inst.N]
    return Instance(A, P2, N2, inst.sigma, inst.op, inst.max_q)


def determine_max_q_per_relation(inst: Instance) -> dict[str, int]:
    result: defaultdict[str, int] = defaultdict(int)

    A = inst.A
    sigma = inst.sigma

    for a in ind(A):
        local_outdegree: defaultdict[str, int] = defaultdict(int)
        for b, r in A.rn_ext[a]:
            local_outdegree[r] += 1

        for rn in sigma.rolenames:
            result[rn] = max(result[rn], local_outdegree[rn])

    return dict(result)


def merge_conj(conj: list[ALCConcept], op: OP) -> ALCConcept:
    if len(conj) == 0 and op == OP.AND:
        return ALCConcept(OP.TOP, "", None, tuple())
    if len(conj) == 0 and op == OP.OR:
        return ALCConcept(OP.BOT, "", None, tuple())

    if len(conj) == 1:
        return conj[0]

    d1 = merge_conj(conj[: len(conj) // 2], op)
    d2 = merge_conj(conj[len(conj) // 2 :], op)
    return ALCConcept(op, name="", value=None, children=(d1, d2))


def deduplicate(lst):
    seen = set()
    res = []
    for a in lst:
        if a not in seen:
            seen.add(a)
            res.append(a)
    return res


def simplify_conj(conj: list[ALCConcept], A: Structure) -> list[ALCConcept]:
    res = []
    for i, c in enumerate(conj):
        if c.operation != OP.CN:
            res.append(c)
            continue

        found = False
        for d in conj[i + 1 :]:
            if d.operation != OP.CN:
                continue
            if A.cn_ext[c.name].issuperset(A.cn_ext[d.name]):
                found = True
                break

        for d in res:
            if d.operation != OP.CN:
                continue
            if A.cn_ext[c.name].issuperset(A.cn_ext[d.name]):
                found = True
                break

        if not found:
            res.append(c)
    return res


def extract_concept(color_register, color_a, color_b, A: Structure) -> ALCConcept:
    assert len(color_register) >= 1
    assert color_a != color_b
    assert color_a in color_register[-1].values()
    assert color_b in color_register[-1].values()

    rev = {id: c for (c, id) in color_register[-1].items()}
    assert len(rev) == len(color_register[-1])
    ca = rev[color_a]
    cb = rev[color_b]

    props = deduplicate(list(ca))

    for c, r in props:
        count_a = list(ca).count((c, r))
        count_b = list(cb).count((c, r))

        if count_a == count_b:
            continue

        if c == -1:
            return ALCConcept(OP.CN, name=r, value=None, children=tuple())

        conj = list()

        for c2, s in cb:
            if s != r or c == c2:
                continue
            d = extract_concept(color_register[0:-1], c, c2, A)
            conj.append(d)

        conj = simplify_conj(conj, A)
        d = merge_conj(deduplicate(conj), OP.AND)

        if count_a > count_b:
            return ALCConcept(OP.GE, name=r, value=count_a, children=(d,))

    # This must terminate as color_a and color_b are guaranteed to be different
    c = extract_concept(color_register, color_b, color_a, A)
    return ALCConcept(OP.NEG, name="", value=None, children=(c,))
