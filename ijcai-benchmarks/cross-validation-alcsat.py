import random
import sys

from alcsat.fitting_alc import FittingALC
from alcsat.fitting_el import determine_relevant_symbols
from alcsat.instance import OP, Instance
from alcsat.preprocessing import ThresholdMethod
from alcsat.structures import structure_from_owl


def chunks(lst: list[int], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def kfold(
    inst: Instance,
    folds: int = 10,
    max_k=10,
    timeout: float = 30,
    tm: ThresholdMethod = ThresholdMethod.INTERVALS,
):
    all_p = list(inst.P)
    all_n = list(inst.N)

    random.shuffle(all_p)
    random.shuffle(all_n)

    # TODO: aufrunden
    p_chunks = list(chunks(all_p, len(all_p) // folds))
    n_chunks = list(chunks(all_n, len(all_n) // folds))

    for i in range(folds):
        this_p = [p for j in range(folds) for p in p_chunks[j] if j != i]
        this_n = [n for j in range(folds) for n in n_chunks[j] if j != i]

        f = FittingALC(inst.A, max_k, this_p, this_n, inst.op, 8, 2, clustering=tm)
        (acc, n, concept) = f.solve_incr_approx(max_k, timeout=timeout)

        other_p = p_chunks[i]
        other_n = n_chunks[i]

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for p in other_p:
            if concept.mc(inst.A, p):
                tp += 1
            else:
                fn += 1
        for n in other_n:
            if concept.mc(inst.A, n):
                fp += 1
            else:
                tn += 1

        acc2 = (tp + tn) / (tp + fn + fp + tn)
        f1 = (2 * tp) / (2 * tp + fp + fn)

        yield (i, concept, acc2, f1)


def sml_benchmark_cross_validate(resultpath: str, tm: ThresholdMethod):
    with open(resultpath, mode="w") as outfile:
        _ = outfile.write("bench, fold, acc, f1, size, evo_size, concept\n")
        for bench in [
            "carcinogenesis",
            "hepatitis",
            "lymphography",
            "mammographic",
            "mutagenesis",
            "nctrer",
            "premierleague",
            "pyrimidine",
            "suramin"
        ]:
            owlfile = f"../sml-benchmarks/{bench}/{bench}.owl"
            pospath = f"../sml-benchmarks/{bench}/full/pos.txt"
            negpath = f"../sml-benchmarks/{bench}/full/neg.txt"

            print(f"== Loading {owlfile}")
            A = structure_from_owl(owlfile)

            P: list[int] = []
            with open(pospath, encoding="UTF-8") as file:
                for line in file:
                    ind = line.rstrip()
                    if ind not in A.indmap:
                        print(
                            f"[ERR] The positive example {ind} does not seem to occur in {owlfile}"
                        )
                        sys.exit(1)
                    P.append(A.indmap[ind])

            N: list[int] = []
            with open(negpath, encoding="UTF-8") as file:
                for line in file:
                    ind = line.rstrip()
                    if ind not in A.indmap:
                        print(
                            f"[ERR] The negative example {ind} does not seem to occur in {owlfile}"
                        )
                        sys.exit(1)
                    N.append(A.indmap[ind])

            sigma = determine_relevant_symbols(A, P + N, 1, 10)
            inst = Instance(
                A,
                P,
                N,
                sigma,
                frozenset(
                    [OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG, OP.LE, OP.GE, OP.DGEQ]
                ),
                2,
            )
            for fold, concept, acc, f1 in kfold(inst, 10, max_k=10, timeout=300, tm=tm):
                _ = outfile.write(
                    f"{bench}, {fold}, {acc}, {f1}, {concept.size()}, {concept.evo_size()}, {concept.to_dl_concept()} \n"
                )
                outfile.flush()


def main():
    sml_benchmark_cross_validate(
        "reproduce-table1-our-tool.txt", ThresholdMethod.INTERVALS
    )

if __name__ == "__main__":
    main()
