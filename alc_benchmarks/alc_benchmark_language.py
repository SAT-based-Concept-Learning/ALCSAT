import os
import sys
import time

import pandas as pd

from alcsat.fitting_alc import FittingALC
from alcsat.instance import ALC_OP
from alcsat.structures import map_ind_name, structure_from_owl

from .alc_benchmark import read_examples_from_json


def benchmark_run(dir):
    cols = ["data set", "t_alcsat", "a_alcsat"]
    data = []
    js_path = None
    kb_path = None
    dsname = None
    for d in filter(lambda x: not x.startswith("."), os.listdir(dir)):
        dataset_dir = os.path.join(dir, d)
        if os.path.isdir(dataset_dir):
            for f in filter(lambda x: not x.startswith("."), os.listdir(dataset_dir)):
                path = os.path.join(dataset_dir, f)
                base, ext = os.path.splitext(path)
                if ext == ".json":
                    js_path = path
                if ext == ".owl":
                    kb_path = path
                dsname = os.path.basename(dataset_dir)
            print(f"Running on {dsname}")
            P, N = read_examples_from_json(js_path)
            A = structure_from_owl(kb_path)

            P = list(map(lambda n: map_ind_name(A, n), P))
            N = list(map(lambda n: map_ind_name(A, n), N))
            start = time.time()
            max_k = 32
            f = FittingALC(A, max_k, P, N, op=ALC_OP)
            a_alcsat, n_alcsat, c_alcsat = f.solve_incr(max_k)
            end = time.time()
            t_alcsat = end - start

            data.append([dsname, t_alcsat, a_alcsat])
            pd.DataFrame(data, columns=cols).to_csv(
                os.path.join(dir, "results_reproduced.csv")
            )


def main():
    benchmark_run(sys.argv[1])


if __name__ == "__main__":
    main()
