from alcsat.preprocessing import ThresholdMethod
import time
from alcsat_cli import L_OP
from alcsat.structures import structure_from_owl
from alcsat.fitting_alc import FittingALC


def main():
    runs = 3
    benchmarks = ["mammographic", "suramin", "mutagenesis"]
    max_k = 8
    intervals = [0, 2, 5, 10, 20, 1000]
    outfile = "reproduce-table2.txt"

    with open(outfile, mode="w") as outfile:
        _ = outfile.write("bench, intervals, accuracy, time\n")
        for benchmark in benchmarks:
            A = structure_from_owl(f"../sml-benchmarks/{benchmark}/{benchmark}.owl")
            pospath = f"../sml-benchmarks/{benchmark}/full/pos.txt"
            negpath = f"../sml-benchmarks/{benchmark}/full/neg.txt"

            P: list[int] = []
            with open(pospath, encoding="UTF-8") as file:
                for line in file.readlines():
                    ind = line.rstrip()
                    P.append(A.indmap[ind])

            N: list[int] = []
            with open(negpath, encoding="UTF-8") as file:
                for line in file.readlines():
                    ind = line.rstrip()
                    N.append(A.indmap[ind])


            for i in intervals:
                times : list[float] = []
                accuracies: list[float] = []
                for run in range(runs):
                    start = time.perf_counter()

                    f = FittingALC(
                        A,
                        max_k,
                        P,
                        N,
                        op=frozenset(L_OP["alcqf"]),
                        workers=8,
                        max_q=2,
                        max_thresholds=i,
                        clustering = ThresholdMethod.INTERVALS

                    )

                    acc, _, _ = f.solve_incr_approx(max_k)

                    end = time.perf_counter()

                    print("==== TOOK {}".format(end - start))
                    accuracies.append(acc)
                    times.append(end - start)

                acc = sum(accuracies) / runs
                t = sum(times) / runs
                outfile.write(f"{benchmark}, {i}, {acc}, {t}\n")
                outfile.flush()
                print(f"Benchmark {benchmark}, Intervals {i} : {acc}, {t}s")


if __name__ == "__main__":
    main()
