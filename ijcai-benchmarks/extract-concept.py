from alcsat.fitting_alc import perfect_fitting
from alcsat.preprocessing import (
    color_refinement,
    ThresholdMethod,
    decode_dataproperties,
)
from alcsat.fitting_el import non_empty_symbols
from alcsat.instance import ALCConcept, OP, Instance
from alcsat.structures import Signature, Structure, structure_from_owl


def main():
    benchmarks = ["carcinogenesis", "mutagenesis", "mammographic", "suramin"]

    resultpath = "outextract2.csv"
    with open(resultpath, mode="w") as outfile:
        _ = outfile.write("bench, method, thresholds, acc, size\n")
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

            sig = non_empty_symbols(A)

            inst = Instance(A, P, N, sig, frozenset(), 2)

            acc, size, c = perfect_fitting(inst, ThresholdMethod.ALL_THRESHOLDS)
            outfile.write(f"{benchmark}, ALL_THRESHOLDS, 0, {acc}, {size}\n")
            # print(c.to_tree())
            print(acc)

            acc, size, c = perfect_fitting(inst, ThresholdMethod.NONE)
            outfile.write(f"{benchmark}, NONE, 0, {acc}, {size}\n")
            print(acc)

            for t in {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}:
                print(f"=== {benchmark} {t} Thresholds")

                acc, size, c = perfect_fitting(inst, ThresholdMethod.INTERVALS, t)
                outfile.write(f"{benchmark}, INTERVALS, {t}, {acc}, {size}\n")
                print(acc)

                acc, size, c = perfect_fitting(inst, ThresholdMethod.KMEANS, t)
                outfile.write(f"{benchmark}, KMEANS, {t}, {acc}, {size}\n")

                acc, size, c = perfect_fitting(
                    inst, ThresholdMethod.NEIGHBOORHOOD_KMEANS, t
                )
                outfile.write(f"{benchmark}, NEIGHBORHOOD_KMEANS, {t}, {acc}, {size}\n")
                outfile.flush()


if __name__ == "__main__":
    main()
