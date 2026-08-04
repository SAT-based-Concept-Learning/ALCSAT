import json
import os
import sys
import time

from ontolearn.heuristics import CELOEHeuristic
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import CELOE, TDL, EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Accuracy
from ontolearn.refinement_operators import ModifiedCELOERefinement
from ontolearn.triple_store import OWLDataHasValue
from ontolearn.utils.static_funcs import compute_f1_score
from owlapy.class_expression import (
    OWLClass,
    OWLClassExpression,
    OWLDataSomeValuesFrom,
    OWLObjectAllValuesFrom,
    OWLObjectComplementOf,
    OWLObjectExactCardinality,
    OWLObjectIntersectionOf,
    OWLObjectMaxCardinality,
    OWLObjectMinCardinality,
    OWLObjectSomeValuesFrom,
    OWLObjectUnionOf,
)
from owlapy.owl_individual import IRI, OWLNamedIndividual
from owlapy.render import DLSyntaxObjectRenderer

# from alc_benchmarks.alc_benchmark import instance_to_dllearner

# from alc_benchmark import instance_to_dllearner


def ontolearn_examples_to_dllearner(kb_path, ont_examples, dest, file_name_prefix):
    with open(ont_examples) as f:
        d = json.load(f)
        for p in d["problems"]:
            pass
            # instance_to_dllearner(
            #     kb_path,
            #     d["problems"][p]["positive_examples"],
            #     d["problems"][p]["negative_examples"],
            #     os.path.join(dest, f"{file_name_prefix}_{p}"),
            # )


def ontolearn_examples_to_flat_json(ont_examples, dest):
    with open(ont_examples) as f:
        d = json.load(f)
        for p in d["problems"]:
            dn = dict()
            dn["P"] = d["problems"][p]["positive_examples"]
            dn["N"] = d["problems"][p]["negative_examples"]
            dn["N_POS"] = len(dn["P"])
            dn["N_NEG"] = len(dn["N"])
            with open(os.path.join(dest, f"ol_ex_fam_rich_{p}.json"), "w+") as f:
                json.dump(dn, f)


def run_celoe(kb_path, P, N):
    start = time.time()
    kb = KnowledgeBase(path=kb_path)
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, P)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, N)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
    end = time.time()
    kb_parse_time = end - start
    print(f"KB parsed after {kb_parse_time} seconds, starting CELOE next.")
    start = time.time()
    qual = Accuracy()
    heur = CELOEHeuristic(
        expansionPenaltyFactor=0.05, startNodeBonus=1.0, nodeRefinementPenalty=0.01
    )
    op = ModifiedCELOERefinement(
        knowledge_base=kb, use_negation=False, use_all_constructor=True
    )

    model = CELOE(
        knowledge_base=kb,
        max_runtime=600,
        refinement_operator=op,
        quality_func=qual,
        heuristic_func=heur,
        max_num_of_concepts_tested=100,
        iter_bound=100,
    )
    model.fit(lp)
    hypotheses = list(model.best_hypotheses(n=3))
    # predictions = model.predict(individuals=list(typed_pos | typed_neg),
    #                           hypotheses=hypotheses)
    prediction = model.best_hypotheses(1, return_node=True)
    rdr = DLSyntaxObjectRenderer()
    end = time.time()
    print(f"Time for running CELOE: {end - start}")
    print(f"Total time: {end - start + kb_parse_time} seconds")
    return prediction.quality, rdr.render(prediction.concept)

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

def run_evo(kb_path, P, N, timeout = 10, card_limit = 3, f1 = True):    
    kb = KnowledgeBase(path=kb_path)    
    
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, P)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, N)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)     
    
    qf = Accuracy()
    if f1:
        qf = F1()

    model = EvoLearner(
        knowledge_base=kb,
        max_runtime=timeout,        
        use_card_restrictions=True,
        use_data_properties=True,   
        card_limit=card_limit,
        quality_func=qf
    )
    model.fit(lp, verbose=True)

    prediction = model.best_hypotheses(1)  
    rdr = DLSyntaxObjectRenderer()    
    c = rdr.render(prediction)
    
    pred = model.best_hypotheses(1, return_node=True)  
    a = accuracy(
         individuals=frozenset({i for i in kb.individuals(prediction)}),
                    pos=lp.pos,
                    neg=lp.neg
    )
    f1 = compute_f1_score(
                    individuals=frozenset({i for i in kb.individuals(prediction)}),
                    pos=lp.pos,
                    neg=lp.neg
                )
    return a, c , owl_concept_size(prediction),f1

def run_tdl(kb_path, P, N, timeout=10):
    kb = KnowledgeBase(path=kb_path)    

    lp = PosNegLPStandard(
                    pos={OWLNamedIndividual(i) for i in P},
                    neg={OWLNamedIndividual(i) for i in N},
                )
    
    # typed_pos = set(map(OWLNamedIndividual, map(IRI.create, P)))
    # typed_neg = set(map(OWLNamedIndividual, map(IRI.create, N)))
    # lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)        
    model = TDL(knowledge_base=kb, max_runtime=timeout, use_nominals=False)
    model.fit(lp)        

    prediction = model.best_hypotheses(1)    
    f1 = compute_f1_score(
                    individuals=frozenset({i for i in kb.individuals(prediction)}),
                    pos=lp.pos,
                    neg=lp.neg
                )
    rdr = DLSyntaxObjectRenderer()    
    c = rdr.render(prediction)
    a = accuracy(frozenset({i for i in kb.individuals(prediction)}), lp.pos, lp.neg)    
    return a,  c , owl_concept_size(prediction), f1



def owl_concept_size(c : OWLClassExpression) -> int:
    if c.is_owl_nothing():
        return 1
    if c.is_owl_thing():
        return 1
    if isinstance(c, OWLClass):
        return 1
    if isinstance(c, OWLNamedIndividual):
        return 1
    if isinstance(c, OWLObjectUnionOf) or isinstance(c, OWLObjectIntersectionOf):
        return 1 + sum(owl_concept_size(d) for d in c.operands())
    if isinstance(c, OWLObjectComplementOf):
        return 1 + owl_concept_size(c.get_operand())
    if isinstance(c, OWLObjectMinCardinality):
        return 1 + owl_concept_size(c.get_filler())
    if isinstance(c, OWLObjectMaxCardinality):
        return 1 + owl_concept_size(c.get_filler())
    if isinstance(c, OWLObjectExactCardinality):
        return 1 + owl_concept_size(c.get_filler())
    if isinstance(c, OWLObjectSomeValuesFrom):
        return 1 + owl_concept_size(c.get_filler())
    if isinstance(c, OWLObjectAllValuesFrom):
        return 1 + owl_concept_size(c.get_filler())
    if isinstance(c, OWLDataHasValue):
        return 1
    if isinstance(c, OWLDataSomeValuesFrom):
        return 1
    print(c)
    return 0


def read_examples_from_json(path):
    with open(path) as f:
        o = json.load(f)
    return o["P"], o["N"]


def main():
    # P,N = read_examples_from_json(sys.argv[2])
    P: list[str] = []
    N: list[str] = []
    with open(sys.argv[2], encoding="UTF-8") as file:
        for line in file:
            ind = line.rstrip()
            P.append(ind)

    with open(sys.argv[3], encoding="UTF-8") as file:
        for line in file:
            ind = line.rstrip()
            N.append(ind)

    q, res = run_evo(sys.argv[1], P, N)
    print(f"{q} {res}")
    # ontolearn_examples_to_dllearner(sys.argv[1], sys.argv[2])
    # ontolearn_examples_to_flat_json(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
