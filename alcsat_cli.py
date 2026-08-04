import argparse
import sys
import time

from alcsat.fitting_alc import FittingALC, FittingMode
from alcsat.instance import OP
from alcsat.structures import structure_from_owl

LANGUAGES = ["el", "eli", "fl0", "ex-or", "all-or", "elu", "alc", "alcq", "alci", "alcqif"]
L_OP = {
    "el": [OP.EX, OP.AND],
    "eli": [OP.EX, OP.AND, OP.INV],
    "fl0": [OP.ALL, OP.AND],
    "ex-or": [OP.EX, OP.OR],
    "all-or": [OP.ALL, OP.OR],
    "elu": [OP.EX, OP.OR, OP.AND],
    "alc": [OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG],
    "alcf": [OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG, OP.DGEQ],
    "alci": [OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG, OP.INV],
    "alcif": [OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG, OP.INV, OP.DGEQ],
    "alcq": [OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG, OP.LE, OP.GE],
    "alcqi": [OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG, OP.LE, OP.GE, OP.INV],
    "alcqif": [OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG, OP.LE, OP.GE, OP.INV, OP.DGEQ],
    "alcqf": [OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG, OP.LE, OP.GE, OP.DGEQ],
}


def main():
    parser = argparse.ArgumentParser(prog="alcsat.py")

    _ = parser.add_argument(
        "kb_owl_file", help="path to a OWL knowledge base in RDF/XML format"
    )
    _ = parser.add_argument(
        "pos_example_list", help="path to a textfile containing positive examples"
    )
    _ = parser.add_argument(
        "neg_example_list", help="path to a textfile containing negative examples"
    )

    _ = parser.add_argument(
        "--language",
        type=str,
        default="alcqf",
        choices=LANGUAGES,
        help="language to learn in, el: {exists, and}, fl0: {forall, and}, ex-or: {exists, or}, all-or: {forall, or}, elu: {exists, and, or}, alc: {forall, exists, and, or, neg}, alcq: {forall, exists, and, or, neg, le, ge} (default=alcq)",
    )

    _ = parser.add_argument("--max_size", type=int, default=12, help="(default=12)")
    _ = parser.add_argument("--max_q", type=int, default=2, help="(default=2)")
    _ = parser.add_argument(
        "--mode",
        choices=[FittingMode.EXACT, FittingMode.APPROX],
        default=FittingMode.APPROX,
        help="(default=approx)",
    )
    _ = parser.add_argument("--notop", action="store_true", help="disables the top concept")
    _ = parser.add_argument("--nobot", action="store_true", help="disables the bottom concept")

    _ = parser.add_argument(
        "--timeout", type=float, default=-1, help="in seconds (default=-1)"
    )

    _ = parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of worker processes (default = 1)",
    )

    args = parser.parse_args()

    owlfile = args.kb_owl_file
    pospath = args.pos_example_list
    negpath = args.neg_example_list

    time_start = time.perf_counter()

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

    time_parsed = time.perf_counter()

    exclude_atomic = []
    if args.notop:
        exclude_atomic.append(OP.TOP)
    if args.nobot:
        exclude_atomic.append(OP.BOT)

    print("== Starting incremental search search for fitting query")
    time_start_solve = time.perf_counter()

    acc = 0
    f = FittingALC(
        A,
        args.max_size,
        P,
        N,
        op=frozenset(L_OP[args.language]),
        workers=args.workers,
        max_q=args.max_q,
        exclude_atomic=exclude_atomic
    )
    remaining_time = -1
    if args.timeout != -1:
        remaining_time = args.timeout - (time.perf_counter() - time_start)

    if args.mode == FittingMode.EXACT:
        acc, _, _ = f.solve_incr(args.max_size, timeout=remaining_time)
    elif args.mode == FittingMode.APPROX:
        acc, _, _ = f.solve_incr_approx(args.max_size, timeout=remaining_time)

    time_solved = time.perf_counter()

    print(
        f"== Took {time_parsed - time_start:.2f}s for reading input and {time_solved - time_start_solve:.3f}s for solving"
    )
    print(f"== Reached accurary {acc:.4f}")


if __name__ == "__main__":
    main()
