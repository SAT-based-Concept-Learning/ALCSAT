import time

from alcsat.fitting_alc import FittingALC
from alcsat.fitting_el import determine_relevant_symbols, non_empty_symbols
from alcsat.instance import OP, Instance
from alcsat.preprocessing import (
    ThresholdMethod,
    bisimulation_reduction,
    encode_dataproperties,
    encode_inverses,
    restrict_neighborhood,
)
from alcsat.structures import structure_from_owl
from alcsat_cli import L_OP


def sizes(A, P, N, max_k, ops, max_q):

    sigma = non_empty_symbols(A)

    inst = Instance(A, P, N, sigma, ops, max_q)

    if OP.INV in inst.op:
        inst, _ = encode_inverses(inst)

    inst.sigma = determine_relevant_symbols(
        inst.A, inst.P + inst.N, 1, max_k - 1
    )

    inst = restrict_neighborhood(inst, max_k)

    if OP.DGEQ in inst.op:
        inst, reverse_data_mapping = encode_dataproperties(
            inst, clustering=ThresholdMethod.INTERVALS, max_k=max_k, max_thresholds=10
        )

    
    size_before = inst.A.max_ind

    inst = bisimulation_reduction(inst, max_k)

    size_after = inst.A.max_ind

    return size_before, size_after

def main():
    runs = 3
    benchmarks = ["mammographic", "hepatitis", "lymphography"]
    max_k = 8
    languages = ["alc", "alcf", "alcqf", "alcqif"]
    workers = 8
    max_q = 2

    outfile = "reproduce-table3.txt"

    with open(outfile, mode="w") as outfile:
        _ = outfile.write("language, benchmark, time_without, time_with, size_without, size_with\n")
        for language in languages:
            for benchmark in benchmarks:

                A = structure_from_owl(f"../sml-benchmarks/{benchmark}/{benchmark}.owl")
                pospath = f"../sml-benchmarks/{benchmark}/full/pos.txt"
                negpath = f"../sml-benchmarks/{benchmark}/full/neg.txt"

                P: list[int] = []
                with open(pospath, encoding="UTF-8") as file:
                    for line in file:
                        ind = line.rstrip()
                        P.append(A.indmap[ind])

                N: list[int] = []
                with open(negpath, encoding="UTF-8") as file:
                    for line in file:
                        ind = line.rstrip()
                        N.append(A.indmap[ind])

                time_with_reduction = []
                time_without_reduction = []

                full_size, reduced_size = sizes(A, P, N, max_k, frozenset(L_OP[language]), max_q)

                for run in range(runs):
                    start = time.perf_counter()

                    f = FittingALC(
                        A,
                        max_k,
                        P,
                        N,
                        op=L_OP[language],
                        workers=workers,
                        max_q=max_q,
                        bisim_reduction=False,
                    )

                    acc, _, _ = f.solve_incr_approx(max_k)

                    end = time.perf_counter()

                    print(f"==== TOOK {end - start}")
                    t1 = end - start
                    time_without_reduction.append(t1)

                    start = time.perf_counter()
                    f = FittingALC(
                        A,
                        max_k,
                        P,
                        N,
                        op=L_OP[language],
                        workers=workers,
                        max_q=max_q,
                        bisim_reduction=True,
                    )

                    acc, _, _ = f.solve_incr_approx(max_k)

                    end = time.perf_counter()

                    print(f"==== TOOK {end - start}")
                    t2 = end - start

                    time_with_reduction.append(t2)

                tw = sum(time_with_reduction) / runs
                two = sum(time_without_reduction) / runs
                outfile.write(f"{language}, {benchmark}, {two}, {tw}, {full_size}, {reduced_size}\n")
                outfile.flush()
                print(f"Language {language} Benchmark {benchmark} : {two}s {tw}s")


if __name__ == "__main__":
    main()
