from ontolearn.metrics import F1
from owlapy.render import DLSyntaxObjectRenderer, ManchesterOWLSyntaxOWLObjectRenderer
from owlapy.owl_individual import OWLNamedIndividual
from ontolearn.learning_problem import PosNegLPStandard
from sklearn.model_selection import StratifiedKFold
from ontolearn.utils.static_funcs import compute_f1_score
import time
from ontolearn.learners import TDL, EvoLearner
from ontolearn.knowledge_base import KnowledgeBase
from alcsat.fitting_el import determine_relevant_symbols
import sys
from alcsat.structures import structure_from_owl
from alcsat.fitting_alc import FittingALC
from alcsat.preprocessing import ThresholdMethod
import random
from alcsat.instance import Instance, OP
import numpy as np
from alc_benchmarks.ontolearn_benchmark import owl_concept_size

def size(concept) -> int:
    rd = ManchesterOWLSyntaxOWLObjectRenderer()
    s = rd.render(concept)
    return len(s.split()) - s.count("[")

def accuracy(individuals, pos, neg):
    tp = 0
    tn = 0

    for p in pos:
        if p in individuals:
            tp += 1

    for n in neg:
        if n not in individuals:
            tn += 1

    return (tp + tn) / (len(pos) + len(neg))


def sml_benchmark_cross_validate(resultpath: str):
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
            "pyrimidine"#,
            #"suramin"
        ]:
            owlfile = f"../sml-benchmarks/{bench}/{bench}.owl"
            pos_path = f"../sml-benchmarks/{bench}/full/pos.txt"
            neg_path = f"../sml-benchmarks/{bench}/full/neg.txt"

            kb = KnowledgeBase(path=owlfile)

            data = dict()

            p: list[str] = []
            with open(pos_path, encoding="UTF-8") as file:
                for line in file.readlines():
                    ind = line.rstrip()
                    p.append(ind)

            n: list[str] = []
            with open(neg_path, encoding="UTF-8") as file:
                for line in file.readlines():
                    ind = line.rstrip()
                    n.append(ind)

            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
            X = np.array(p + n)
            y = np.array([1.0 for _ in p] + [0.0 for _ in n])

            for ith, (train_index, test_index) in enumerate(kf.split(X, y)):
                #
                data.setdefault("LP", []).append(owlfile)
                data.setdefault("Fold", []).append(ith)
                # () Extract positive and negative examples from train fold
                train_pos = {
                    pos_individual
                    for pos_individual in X[train_index][y[train_index] == 1]
                }
                train_neg = {
                    neg_individual
                    for neg_individual in X[train_index][y[train_index] == 0]
                }

                # Sanity checking for individuals used for training.
                assert train_pos.issubset(p)
                assert train_neg.issubset(n)

                # () Extract positive and negative examples from test fold
                test_pos = {
                    pos_individual
                    for pos_individual in X[test_index][y[test_index] == 1]
                }
                test_neg = {
                    neg_individual
                    for neg_individual in X[test_index][y[test_index] == 0]
                }

                # Sanity checking for individuals used for testing.
                assert test_pos.issubset(p)
                assert test_neg.issubset(n)

                train_lp = PosNegLPStandard(
                    pos={OWLNamedIndividual(i) for i in train_pos},
                    neg={OWLNamedIndividual(i) for i in train_neg},
                )

                test_lp = PosNegLPStandard(
                    pos={OWLNamedIndividual(i) for i in test_pos},
                    neg={OWLNamedIndividual(i) for i in test_neg},
                )

                start_time = time.time()
                evolearner = EvoLearner(
                    knowledge_base=KnowledgeBase(path=owlfile),
                    quality_func=F1(),
                    max_runtime=300,
                    card_limit=3
                )
                pred_evo = evolearner.fit(train_lp).best_hypotheses()
                rt_evo = time.time() - start_time

                # () Quality on the training data
                train_f1_evo = compute_f1_score(
                    individuals=frozenset({i for i in kb.individuals(pred_evo)}),
                    pos=train_lp.pos,
                    neg=train_lp.neg,
                )
                # () Quality on test data
                test_f1_evo = compute_f1_score(
                    individuals=frozenset({i for i in kb.individuals(pred_evo)}),
                    pos=test_lp.pos,
                    neg=test_lp.neg,
                )
                test_acc = accuracy(frozenset({i for i in kb.individuals(pred_evo)}), test_lp.pos, test_lp.neg)


                render = DLSyntaxObjectRenderer()

                _ = outfile.write(
                    f"{bench}, {ith}, {test_acc}, {test_f1_evo}, {owl_concept_size(pred_evo)}, {size(pred_evo)}, {render.render(pred_evo)} \n"
                )
                outfile.flush()

                data.setdefault("Train-F1-Evo", []).append(train_f1_evo)
                data.setdefault("Test-F1-Evo", []).append(test_f1_evo)
                data.setdefault("RT-TDL", []).append(rt_evo)
                print(f"Evo Train Quality: {train_f1_evo:.3f}", end="\t")
                print(f"Evo Test Quality: {test_f1_evo:.3f}", end="\t")
                print(f"Evo Runtime: {rt_evo:.3f}")


def main():
    sml_benchmark_cross_validate("reproduce-table1-evolearner.txt")


if __name__ == "__main__":
    main()
