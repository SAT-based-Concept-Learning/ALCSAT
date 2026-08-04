import time
from alcsat_cli import L_OP
from alcsat.structures import structure_from_owl
from alcsat.fitting_alc import FittingALC


def main():
    workers = [1, 2, 3, 4, 5, 6, 7, 8]
    runs = 3
    benchmarks = ["mammographic", "carcinogenesis", "lymphography"]
    a = []
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

        times: dict[int, list[float]] = {}

        for w in workers:
            times[w] = []
            for run in range(runs):
                start = time.perf_counter()

                f = FittingALC(
                    A,
                    8,
                    P,
                    N,
                    op=L_OP["alc"],
                    workers=w,
                    max_q=2,
                )

                acc, _, _ = f.solve_incr_approx(8)

                end = time.perf_counter()

                print("==== TOOK {}".format(end - start))
                times[w].append(end - start)

        for w in workers:
            print(f"Benchmark {benchmark}, Worker {w} : {sum(times[w]) / runs}s")
            a.append(f"Benchmark {benchmark}, Worker {w} : {sum(times[w]) / runs}s")
    for x in a:
        print(x)

if __name__ == "__main__":
    main()
