import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from owlready2 import default_world, get_ontology
from rdflib import Graph

from alcsat.benchmark_tools import construct_owl_from_structure
from alcsat.fitting_alc import FittingALC
from alcsat.instance import ALC_OP, OP
from alcsat.preprocessing import (
    color_refinement,
    restrict_to_neighborhood,
)
from alcsat.structures import Signature, Structure, map_ind_name, structure_from_owl

from .ontolearn_benchmark import run_evo, run_tdl

CELOE_PATH = ""
SPARCEL_PATH = ""

RANDOM_SEED = 1

QALL = """
SELECT DISTINCT ?0 WHERE {
    ?0 a <http://www.w3.org/2002/07/owl#NamedIndividual>.
    }
    """

random.seed(RANDOM_SEED)


def query_and_print(path, query):
    g = Graph()
    with open(path, "r") as f:
        g.parse(f, format="application/rdf+xml")
    qres = g.query(query)
    for res in list(qres):
        print(res[0].toPython())


def query_for_examples(kb_path, q_pos, q_neg, n_pos, n_neg):
    g = get_ontology(kb_path).load()
    p_res = list(map(lambda x: x[0].get_iri(), default_world.sparql(q_pos)))
    P = random.sample(p_res, n_pos)
    n_res = list(map(lambda x: x[0].get_iri(), default_world.sparql(q_neg)))
    N = random.sample(n_res, n_neg)
    return P, N


def read_examples_from_json(path):
    with open(path) as f:
        o = json.load(f)
    return o["P"], o["N"]


def query_and_save(path, q_pos, q_neg, n_pos, n_neg, dest_path, filename):
    d = dict()
    P, N = query_for_examples(path, q_pos, q_neg, n_pos, n_neg)
    d["Q_POS"] = q_pos
    d["Q_NEG"] = q_pos
    d["N_POS"] = len(P)
    d["N_NEG"] = len(N)
    d["P"] = P
    d["N"] = N
    with open(os.path.join(f"{dest_path}", f"{filename}.json"), "w+") as f:
        json.dump(d, f, indent=4)


def query_and_solve(path, q_pos, q_neg, n_pos, n_neg, k):
    P, N = query_for_examples(path, q_pos, q_neg, n_pos, n_neg)
    A = structure_from_owl(path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A, k, P, N, op=ALC_OP)
    return f.solve()


def instance_to_sparcel(kb_path, p, n, dest, file_name="dl_instance"):
    file = os.path.join(dest, f"{file_name}.conf")
    with open(file, "w+", encoding="utf-8") as f:
        f.write('ks.type = "OWL File"\n')
        f.write(f'ks.fileName = "{os.path.relpath(kb_path)}"\n')
        f.write('reasoner.type = "fast instance checker"\n')
        f.write("reasoner.sources = { ks }\n")
        f.write('lp.type = "org.dllearner.algorithms.ParCEL.ParCELPosNegLP"\n')
        k = ",".join([f'"{x}"' for x in p if x.isascii()])
        f.write(f"lp.positiveExamples = {{ {k} }}\n")
        k = ",".join([f'"{x}"' for x in n if x.isascii()])
        # k = ",".join(map(lambda x : f'"{x}"',n))
        f.write(f"lp.negativeExamples = {{ {k} }}\n")
        f.write(
            'algorithm.type = "org.dllearner.algorithms.ParCELEx.ParCELearnerExV2"\n'
        )
        f.write("algorithm.maxExecutionTimeInSeconds = 60\n")
        f.write("algorithm.numberOfWorkers = 1\n")
        f.write("algorithm.splitter = splitter\n")
        f.write(
            'splitter.type = "org.dllearner.algorithms.ParCEL.split.ParCELDoubleSplitterV1"\n'
        )
        # f.write('alg.writeSearchTree = true\n')


def run_sparcel(kb_pth, ex_path, celoe_path):
    P, N = read_examples_from_json(ex_path)
    instance_to_sparcel(kb_pth, P, N, ".", "sparcel_instance")
    outpt = subprocess.check_output(
        ["java", "-jar", SPARCEL_PATH, "./sparcel_instance.conf"]
    )
    output = str(outpt)
    lines = output.split("\\n")

    query = lines[-7]

    #    query_red = query[8:]
    print(query)

    print(lines[-5:-1])

    tp = int(lines[-5].split(":")[1])
    fp = int(lines[-4].split(":")[1])
    tn = int(lines[-3].split(":")[1])
    fn = int(lines[-2].split(":")[1])

    return (tp + tn) / (tp + fp + tn + fn), query


def run_celoe(kb_path, ex_path):
    P, N = read_examples_from_json(ex_path)
    confpath = os.path.join(os.path.dirname(ex_path), "dllearner_instance.conf")
    instance_to_dllearner(kb_path, P, N, os.path.dirname(ex_path), "dllearner_instance")
    outpt = subprocess.check_output([CELOE_PATH, confpath])
    output = str(outpt)
    lines = output.split("\\n")
    i = 0
    for l in lines:
        if l == "solutions:":
            break
        i += 1
    return float(
        lines[i + 1][lines[i + 1].find("pred. acc.:") + 12 : lines[i + 1].find("%, F")]
    )


def solve_fixed_k(path, ex_path, k):
    A = structure_from_owl(path)
    P, N = read_examples_from_json(ex_path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A, k, P, N, op=ALC_OP)
    return f.solve()


def solve(path, ex_path, k):
    A = structure_from_owl(path)
    P, N = read_examples_from_json(ex_path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A, k, P, N, op=ALC_OP)
    return f.solve_incr(k)


# def solve_old(path, ex_path, k):
#     A = structure_from_owl(path)
#     P,N = read_examples_from_json(ex_path)
#     P = list(map(lambda n: map_ind_name(A, n), P))
#     N = list(map(lambda n: map_ind_name(A, n), N))
#     f = fitting_alc1.FittingALC(A,k,P,N, op = {EX,ALL,OR,AND, NEG})
#     return f.solve()


def test(path, P, N):
    A = structure_from_owl(path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A, 6, P, N, op=ALC_OP)
    f.solve()


def run_on_ontolearn_examples(kb_path, json_path, problem_key, k):
    A = structure_from_owl(kb_path)
    with open(json_path) as f:
        d = json.load(f)
    P = d["problems"][problem_key]["positive_examples"]
    N = d["problems"][problem_key]["negative_examples"]
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A, k, P, N, op=ALC_OP)
    f.solve_incr(k)


def instance_to_dllearner(kb_path, p, n, dest, file_name="dl_instance"):
    file = os.path.join(dest, f"{file_name}.conf")
    with open(file, "w+") as f:
        f.write('ks.type = "OWL File"\n')
        f.write(f'ks.fileName = "{kb_path}"\n')
        f.write('measure.type = "gen_fmeasure"')
        f.write('reasoner.type = "closed world reasoner"\n')
        f.write("reasoner.sources = { ks }\n")
        f.write('lp.type = "posNegStandard"\n')
        k = "{" + ",".join(map(lambda x: f'"{x}"', p)) + "}"
        f.write(f"lp.positiveExamples = {k}\n")
        k = "{" + ",".join(map(lambda x: f'"{x}"', n)) + "}"
        f.write(f"lp.negativeExamples = {k}\n")
        f.write('alg.type = "celoe"\n')
        f.write("alg.maxExecutionTimeInSeconds = 300\n")
        f.write("alg.writeSearchTree = false\n")
        f.write('h.type ="celoe_heuristic"\n')
        f.write("h.expansionPenaltyFactor = 0.02\n")
        f.write("alg.stopOnFirstDefinition = true\n")


#        f.write('alg.maxNrOfResults = 1\n')
# f.write('useMinimizer = false\n')
# alg.noisePercentage = 32
# //alg.maxClassDescriptionTests = 10000000


def json_to_dllearner(kb_path, json_path, dest_dir):
    with open(json_path) as f:
        d = json.load(f)
        instance_to_dllearner(kb_path, d["P"], d["N"], dest_dir, Path(json_path).stem)

def json_to_txt(json_path):        
    with open(json_path, 'r') as f:
        d = json.load(f)
        write_examples(d["P"], d["N"],os.path.join(os.path.dirname(json_path)))        
    with open(json_path, 'w') as f:
        json.dump(d,f, indent = 4)

def jsons_to_dllearner(kb_path, dir, dest_dir):
    for file in os.listdir(dir):
        if os.path.splitext(file)[1] == ".json":
            json_to_dllearner(kb_path, os.path.join(dir, file), dest_dir)

def reduce_size_by_examples(kb_path, json_path, newpath, filename, k):
    P, N = read_examples_from_json(json_path)
    A = structure_from_owl(kb_path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    B, m = restrict_to_neighborhood(k - 1, A, P + N)
    construct_owl_from_structure(os.path.join(newpath, f"{filename}.owl"), B)


def examples_by_queries(
    kb_path,
    queries_path,
    q_pos,
    q_neg,
    n_pos,
    n_neg,
    dest_dir,
    file_name,
    random_pos=True,
    random_neg=True,
    exclude_pos_from_neg=False,
):
    g = get_ontology(kb_path).load()
    graph = default_world.as_rdflib_graph()
    d = dict()
    with open(queries_path, "r") as f:
        dq = json.load(f)
    d["q_pos"] = dq[q_pos]
    if q_neg is not None:
        d["q_neg"] = dq[q_neg]
    else:
        d["q_neg"] = "complement"
    p_res = list(
        map(lambda x: x[0].get_iri(), default_world.sparql(dq[q_pos]["SPARQL"]))
    )
    # p_res = list(map(lambda x : x[0].get_iri(),list(  graph.query_owlready(dq[q_pos]["SPARQL"]))))
    print(f"Positive:{len(p_res)}")
    if not p_res or n_neg > len(p_res):
        return False
    if random_pos:
        P = random.sample(p_res, n_pos)
    else:
        P = p_res[:n_pos]

    if q_neg is not None:
        # n_res_r = list( map(lambda x : x[0].get_iri(),list(graph.query_owlready(dq[q_neg]["SPARQL"]))))
        n_res_r = list(
            map(lambda x: x[0].get_iri(), default_world.sparql(dq[q_neg]["SPARQL"]))
        )
    else:
        n_res_r = list(map(lambda x: x[0].get_iri(), default_world.sparql(QALL)))
    n_res = []
    if q_neg is None or exclude_pos_from_neg:
        for e in n_res_r:
            if e not in p_res:
                n_res.append(e)
    else:
        n_res = n_res_r
    print(f"Negative:{len(n_res)}")
    if not n_res or n_neg > len(n_res):
        return False
    if random_neg:
        N = random.sample(n_res, n_neg)
    else:
        N = n_res[:n_pos]
    d["n_pos"] = len(P)
    d["n_neg"] = len(N)
    d["P"] = P
    d["N"] = N
    d["rnd_pos"] = random_pos
    d["rnd_neg"] = random_neg
    d["random_seed"] = RANDOM_SEED
    with open(os.path.join(dest_dir, file_name), "w+") as f:
        json.dump(d, f)
    return True


def benchmark_run(dir):
    cols = [
        "data set",
        "t_celoe",
        "t_evo",
        "t_spacel",
        "t_alcsat",
        "a_celoe",
        "a_evo",
        "a_spacel",
        "a_alcsat",
    ]
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

            start = time.time()
            a_celoe = run_celoe(kb_path, js_path)
            end = time.time()
            t_celoe = end - start

            start = time.time()
            a_evo, c_evo = run_evo(kb_path, P, N)
            end = time.time()
            t_evo = end - start

            start = time.time()
            a_sparcel, c_sparcel = run_sparcel(kb_path, js_path)
            end = time.time()
            t_sparcel = end - start

            P = list(map(lambda n: map_ind_name(A, n), P))
            N = list(map(lambda n: map_ind_name(A, n), N))
            start = time.time()
            max_k = 32
            f = FittingALC(A, max_k, P, N, op=ALC_OP)
            a_alcsat, n_alcsat, c_alcsat = f.solve_incr(max_k)
            end = time.time()
            t_alcsat = end - start

            data.append(
                [
                    dsname,
                    t_celoe,
                    t_evo,
                    t_sparcel,
                    t_alcsat,
                    a_celoe,
                    a_evo,
                    a_sparcel,
                    a_alcsat,
                ]
            )
            pd.DataFrame(data, columns=cols).to_csv(
                os.path.join(dir, "results_reproduced.csv")
            )


def benchmark_gen(
    kb_path, queries_path, dest_dir, q_ind, n_pos, n_neg, complement_for_neg=False
):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    red_kb_path_filename = f"yago_family_reduced_kb_{q_ind}_({n_pos},{n_neg})"
    js_path = os.path.join(dest_dir, f"{red_kb_path_filename}.json")
    q_pos = f"Q_p{q_ind}"
    if not complement_for_neg:
        q_neg = f"Q_n{q_ind}"
    else:
        q_neg = None
    if not os.path.exists(js_path):
        if not examples_by_queries(
            kb_path, queries_path, q_pos, q_neg, n_pos, n_neg, dest_dir, js_path
        ):
            return
    reduce_size_by_examples(kb_path, js_path, dest_dir, red_kb_path_filename, 5)


def kCrossVal(P, N, k):
    def toPN(Sp):
        Pp = list(filter(lambda x: x[1] == 1, Sp))
        Nn = list(filter(lambda x: x[1] == 0, Sp))
        return Pp, Nn

    n = len(P) + len(N)
    S = [(x, 0) for x in N] + [(x, 1) for x in P]
    random.shuffle(S)
    subsamples = []
    i = 0
    for i in range(k - 1):
        subsamples.append(S[i : i + (n // k)])
        i += n // k
    subsamples.append(S[i:])
    for i in range(k):
        yield (
            toPN(sum(subsamples[0:i]) + sum(subsamples[i + 1 : k + 1])),
            toPN(subsamples[i]),
        )


def to_tex(path):
    df = pd.read_csv(path)
    df.style.format(decimal=",", thousands=".", precision=2).to_latex(
        os.path.join(os.path.dirname(path), "tex.txt")
    )


# 1: f"yago_family_m_and_f_{k}-descendant_p{i}-n{i}"
# 2: f"yago_family_and_paths_d3_8-{k}"


def benchmark_gen_t():
    if not os.path.exists(sys.argv[3]):
        os.mkdir(sys.argv[3])
    i = 200
    for k in [6]:  # range(4,5):
        dest_dir = os.path.join(sys.argv[3], f"yago_language_{k}_{i}")
        benchmark_gen(sys.argv[1], sys.argv[2], dest_dir, k, i, i)


def convertToTikzCsv(files):
    compare_cols = ["t_celoe", "t_evo", "t_spacel"]
    result_csv = [
        "family_celoe_alcsat_time.csv",
        "family_evo_alcsat_time.csv",
        "family_sparcel_alcsat_time.csv",
    ]

    nd = [None, None, None]
    for f in files:
        d = pd.read_csv(f)
        for i, c in enumerate(compare_cols):
            if nd[i] is None:
                nd[i] = d[[c, "t_alcsat"]]
            else:
                nd[i] = pd.concat([nd[i], d[[c, "t_alcsat"]]])
    for i in range(3):
        nd[i].to_csv(result_csv[i], index=False)


def convertToTikzCsvTwoFiles(files):
    compare_col = "t_alcsat"  #'t_evo' 't_sparcel'
    result_csv = "family_alcsat_alcsat+_time.csv"  # "family_celoe_alcsat_time.csv" #"family_evo_alcsat_time.csv" "family_sparcel_alcsat_time.csv"

    if len(files) % 2 == 0:
        nd = None
        i = 0
        while i < len(files):
            d1 = pd.read_csv(files[i])
            d2 = pd.read_csv(files[i + 1]).rename(columns={"t_alcsat": "t_alcsat+"})
            d = d1[["t_alcsat"]]
            d = pd.concat([d, d2[["t_alcsat+"]]], axis=1)
            if nd is None:
                nd = d
            else:
                nd = pd.concat([nd, d])
            i += 2
        nd.to_csv(result_csv, index=False)


def convertCsv(files):
    compare_cols = ["t_celoe", "t_evo", "t_spacel"]
    result_csv = [
        "family_celoe_alcsat_time.csv",
        "family_evo_alcsat_time.csv",
        "family_sparcel_alcsat_time.csv",
    ]

    time_intervals = range(0, 241, 20)

    data = pd.DataFrame(
        index=time_intervals, columns=["n_alcsat", "n_evo", "n_sparcel", "n_celoe"]
    )

    nd = None
    for f in files:
        d = pd.read_csv(f)
        if nd is None:
            nd = d
        else:
            nd = pd.concat([nd, d])
    for t in time_intervals:
        n_alcsat = 0
        n_celoe = 0
        n_sparcel = 0
        n_evo = 0
        for i in range(nd.shape[0]):
            if nd.iloc[i]["t_alcsat"] <= t and nd.iloc[i]["a_alcsat"] == 1.0:
                n_alcsat += 1
            if nd.iloc[i]["t_celoe"] <= t and nd.iloc[i]["a_celoe"] == 100.0:
                n_celoe += 1
            if nd.iloc[i]["t_spacel"] <= t and nd.iloc[i]["a_spacel"] == 1.0:
                n_sparcel += 1
            if nd.iloc[i]["t_evo"] <= t and nd.iloc[i]["a_evo"] == 1.0:
                n_evo += 1
        data.at[t, "n_alcsat"] = n_alcsat
        data.at[t, "n_celoe"] = n_celoe
        data.at[t, "n_sparcel"] = n_sparcel
        data.at[t, "n_evo"] = n_evo
    data.to_csv("data_graph.csv")

def components(v : set[int], e : set[int,int]):
    components = []
    while len(v) != 0:
        c = set()
        x = v.pop()
        c.add(x)
        change = True
        while change:    
            change = False
            for y,z in e:                
                if y in c:
                    if z in v:                
                        v.remove(z)
                    if z not in c:
                        c.add(z)
                        change = True
        components.append(c)
    return components

def alcq_examples_from_bisim(A: Structure, n_ex=10):
    sr = set([t[1] for s in A.rn_ext.values() for t in s])
    sigma = Signature(A.cn_ext.keys(), sr)
    color_alc, _ = color_refinement(A, sigma, False, -1)
    color_alcq, _ = color_refinement(A, sigma, True, -1)
    classes_alc: defaultdict[int, list[int]] = defaultdict(list)
    #classes_alcq: defaultdict[int, list[int]] = defaultdict(list)
    for a in range(A.max_ind):
        classes_alc[color_alc[a]].append(a)
    #    classes_alcq[color_alcq[a]].append(a)

    classes = list(classes_alc.keys())

    random.shuffle(classes)

    exs = []
    # for k in classes:
    #     v = classes_alc[k]
    #     for a in v:
    #         e = []
    #         for b in v:
    #             if a != b and color_alcq[a] != color_alcq[b]:
    #                 e.append(b)
    #         if len(e) != 0:
    #             if pos_len != -1 and len(e) > pos_len:
    #                 e = random.sample(e, pos_len)
    #             yield e, [a]

    for i in range(0,len(classes),2):
    #for k in classes:        
        v = classes_alc[classes[i]] + classes_alc[classes[i+1]]
        vert = set(v)
        edg = set([(x,y) for x in v for y in v if color_alcq[x] == color_alcq[y]])        
        comp = components(vert, edg)    
        print(len(comp))
        P = [x.pop() for i,x in enumerate(comp[:len(comp)//2]) if i < n_ex]        
        N = [x.pop() for i,x in enumerate(comp[len(comp)//2:]) if i < n_ex]        
        if len(P) > 0 and len(N) > 0:
            yield P,N

def reduce_size_by_examples2(A: Structure, P, N, k, dest=None):    
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    B, m = restrict_to_neighborhood(k - 1, A, P + N)
    if dest:
        construct_owl_from_structure(dest, B)
    return B


def write_examples(P, N, path):
    with open(os.path.join(path, "pos.txt"), "w") as f:
        f.writelines(map(lambda x: f"{x}\n", P))
    with open(os.path.join(path, "neg.txt"), "w") as f:
        f.writelines(map(lambda x: f"{x}\n", N))


def examples_from_bisim(kb_path, output_dir, n_ex = 10, max_datasets = -1):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    A = structure_from_owl(kb_path)    
    ind_map_inv = {v: k for k, v in A.indmap.items()}  
    i=1
    for P, N in alcq_examples_from_bisim(A, n_ex=n_ex):        
        if i > max_datasets and not max_datasets == -1:
            break
        #f = FittingALC(
        #    A,
        #    12,
        #    P,
        #    N,
        #    op=frozenset([OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG, OP.LE, OP.GE]),
        #    workers=10,
        #    max_q=5
        #)
        #a, k, sol = f.solve_incr(12)
        #if a > 0 and k > 5:
        P_s = [ind_map_inv[x] for x in P]
        N_s = [ind_map_inv[x] for x in N]
        dest_dir = os.path.join(output_dir, f"{len(P_s)+len(N_s)}ex-{i!s}")
        os.mkdir(dest_dir)
        write_examples(P_s, N_s, dest_dir)
        reduce_size_by_examples2(
            A, P_s, N_s, 16, dest=os.path.join(dest_dir, "kb_reduced.owl")
        )
        #with open(os.path.join(dest_dir, "fitting_concept.txt"), "w") as f:
            #f.write(sol.to_tree())            
        i+=1

def read_examples(path):
        P,N = [],[]
        with open(os.path.join(path, "pos.txt")) as f:
            P = list(map(lambda s: s.rstrip(),f.readlines()))
        with open(os.path.join(path, "neg.txt")) as f:
            N = list(map(lambda s: s.rstrip(),f.readlines()))
        return P,N

def examples_from_bisim_run(dir_path, overwrite_all = False):
    for d in filter(lambda s : not s.startswith(".") and os.path.isdir(os.path.join(dir_path,s)) ,os.listdir(dir_path)):   
        json_path = os.path.join(dir_path, d, "results.json")
        jd = dict()
        if os.path.exists(json_path) and not overwrite_all:
            with open(json_path, 'r') as f:
                jd = json.load(f)

        kb_path = os.path.join(dir_path,d,"kb_reduced.owl")        
        P,N = read_examples(os.path.join(dir_path,d))

        if not "TDL" in jd.keys() or overwrite_all:
            try:
                start = time.time()
                a, c, c_size, f1 = run_tdl(kb_path,P,N, timeout=300) 
                end = time.time()
                jd["TDL"] = {"concept" : c, "size" : c_size, "accuracy" : a, "time" : end-start, "f1" : f1}
            except ValueError:
                jd["TDL"] = {"concept" : "None", "size" : -1, "accuracy" : -1, "time" : -1, "f1" : -1}

        if not "Evolearner" in jd.keys() or overwrite_all:
            start = time.time()
            a, c, c_size, f1 = run_evo(kb_path,P,N, timeout=300, card_limit=3, f1 = True)
            end = time.time()
            jd["Evolearner"] = {"concept" : c, "size" : c_size, "accuracy" : a, "time" : end-start, "f1" : f1}
            # f1, _,_ = run_evo(kb_path,P,N, timeout=300, card_limit=3, f1 = True)
            # jd["Evolearner"] = {"concept" : c, "size" : c_size, "accuracy" : a, "time" : end-start, "f1" : f1 }
        
        if not "ALCSAT" in jd.keys() or overwrite_all:
            start = time.time()
            A = structure_from_owl(kb_path)
            P_i = [A.indmap[x] for x in P]
            N_i = [A.indmap[x] for x in N]
            f = FittingALC(A, 16, P_i,N_i, op=frozenset([OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG, OP.LE, OP.GE]),workers=8, max_q=3)
            a, c_size, c = f.solve_incr_approx(16,timeout=300)
            end = time.time()
            tp = 0
            fp = 0
            tn = 0
            fn = 0

            for p in P_i:
                if c.mc(A, p):
                    tp += 1
                else:
                    fn += 1
            for n in N_i:
                if c.mc(A, n):
                    fp += 1
                else:
                    tn += 1

            f1 = (2 * tp) / (2 * tp + fp + fn)
            
            
            jd["ALCSAT"] = {"concept" : c.to_dl_concept(), "size" : c_size, "accuracy" : a, "time" : end-start, "f1":f1}

        f = open(json_path, 'w')
        json.dump(jd, f, indent=4)
        f.close()

def examples_from_bisim_rerunevo(dir_path, overwrite_all = False):
    for d in filter(lambda s : not s.startswith(".") and os.path.isdir(os.path.join(dir_path,s)) ,os.listdir(dir_path)):   
        json_path = os.path.join(dir_path, d, "results.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                jd = json.load(f)

        kb_path = os.path.join(dir_path,d,"kb_reduced.owl")
        P,N = read_examples(os.path.join(dir_path,d))
                
        # start = time.time()                
        # f1, c,c_size = run_evo(kb_path,P,N, timeout=300, card_limit=3, f1 = True)
        # end = time.time()
        start = time.time()
        a, c, c_size, f1 = run_evo(kb_path,P,N, timeout=300, card_limit=3)
        end = time.time()
        jd["EvolearnerF1"] = {"concept" : c, "size" : c_size, "accuracy" : a, "time" : end-start, "f1" : f1}
        #jd["EvolearnerF1"] = {"concept" : c, "size" : c_size, "f1" : f1,"time" : end-start}                
        

        f = open(json_path, 'w')
        json.dump(jd, f, indent=4)
        f.close()

def alcq_benchmarks_to_csv(dir_path):
    rows = []
    dirs = sorted(filter(lambda s : not s.startswith(".") and os.path.isdir(os.path.join(dir_path,s)) ,os.listdir(dir_path)),reverse = True)
    avg_alcsat = defaultdict(int)
    avg_evo = defaultdict(int)
    avg_tdl = defaultdict(int)
    avg_f1_alcsat = defaultdict(int)
    avg_f1_evo = defaultdict(int)
    avg_f1_tdl = defaultdict(int)
    n_ex_sets_tdl = defaultdict(int)
    n_ex_sets = defaultdict(int) 
    for d in dirs:            
        P,N = read_examples(os.path.join(dir_path,d))
        m = len(P) + len(N)
        os.path.join(dir_path, d, "results.json")
        f = open(os.path.join(dir_path, d, "results.json"),'r')
        d = json.load(f)
        avg_alcsat[m] += d["ALCSAT"]["accuracy"]
        avg_f1_alcsat[m] += d["ALCSAT"]["f1"]
        avg_evo[m] += d["Evolearner"]["accuracy"]
        avg_f1_evo[m] += d["Evolearner"]["f1"]
        if d["TDL"]["concept"] != "None":
            avg_tdl[m] += d["TDL"]["accuracy"]
            avg_f1_tdl[m] += d["TDL"]["f1"]
            n_ex_sets_tdl[m] +=1
        n_ex_sets[m] += 1
        rows.append([m, d["ALCSAT"]["accuracy"], d["Evolearner"]["accuracy"],d["TDL"]["accuracy"], d["ALCSAT"]["time"], d["Evolearner"]["time"],d["TDL"]["time"],d["ALCSAT"]["size"],d["Evolearner"]["size"],d["TDL"]["size"],d["ALCSAT"]["f1"],d["Evolearner"]["f1"],d["TDL"]["f1"]])
        f.close()

    df_avg_val = []
    for m in n_ex_sets.keys():
        if n_ex_sets_tdl[m] > 0:
            df_avg_val.append([m,avg_alcsat[m]/n_ex_sets[m],avg_evo[m]/n_ex_sets[m], avg_tdl[m]/n_ex_sets_tdl[m], avg_f1_alcsat[m]/n_ex_sets[m],avg_f1_evo[m]/n_ex_sets[m], avg_f1_tdl[m]/n_ex_sets_tdl[m]])
        else:
            df_avg_val.append([m,avg_alcsat[m]/n_ex_sets[m],avg_evo[m]/n_ex_sets[m], -1,avg_f1_alcsat[m]/n_ex_sets[m],avg_f1_evo[m]/n_ex_sets[m], -1])

    df_avg = pd.DataFrame(df_avg_val, columns = ['m', 'a_avg_alcsat', 'a_avg_evo', 'a_avg_tdl','f_avg_alcsat', 'f_avg_evo', 'f_avg_tdl'])
    df_avg.to_csv(os.path.join(dir_path, "data_avg.csv"), index = False)

    df = pd.DataFrame(rows,columns = ['m', 'a_alcsat', 'a_evo', 'a_tdl', 't_alcsat', 't_evo', 't_tdl', 's_alcsat', 's_evo', 's_tdl','f_alcsat', 'f_evo', 'f_tdl'])
    df.to_csv(os.path.join(dir_path, "data.csv"), index = False)


def alcq_combine_csvs(path1, path2, out_path):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    
    data = []
    for index in set(df1.index.union(df2.index)):                
        if index in df1.index and index in df2.index:
            data.append([index, (df1.loc[index, 'a_avg_alcsat'] + df2.loc[index, 'a_avg_alcsat'])/2 ])
        elif index in df1.index:
            data.append([index, df1.loc[index, 'a_avg_alcsat']])        
        elif index in df2.index:
            data.append([index, df2.loc[index, 'a_avg_alcsat']])        

    
    data_cols = ["m", "a_avg_alcsat"]#, "a_avg_evo", "a_avg_tdl"]

    df = pd.DataFrame(data, columns= data_cols)

    # Write CSV
    df.to_csv(out_path, index=False)


def test_evo_data_properties_write_file():
    kb_path = os.path.join(os.path.dirname(__file__), "tmp.owl")
    A = Structure(4, {"A" : {0,1,2,3}}, {0 : {(1,'r')},1 : {},2 : {(3,'r')}, 3 : {}}, {0 : [], 1 : [(2.0, "http://www.w3.org/2001/XMLSchema#double", "T")], 2 : [], 3 : [(40.0, "http://www.w3.org/2001/XMLSchema#double", "T")]}, {
        'http://yago-knowledge.org/resource/ʻElisiva_Fusipala_Taukiʻonetuku' : 0,
        'http://yago-knowledge.org/resource/ʻAnaseini_Takipō' : 1,
        'http://yago-knowledge.org/resource/ʻEtuate_Lavulavu' : 2,
        'http://yago-knowledge.org/resource/ʻAkosita_Lavulavu' : 3
    } ,{None: 'http://www.w3.org/2002/07/owl#', 'owl': 'http://www.w3.org/2002/07/owl#', 'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'xsd': 'http://www.w3.org/2001/XMLSchema#', 'rdfs': 'http://www.w3.org/2000/01/rdf-schema#', 'shacl': 'http://www.w3.org/ns/shacl#', 'schema': 'http://schema.org/', 'schema1': 'http://yago-knowledge.org/schema#'})
    #construct_owl_from_structure(kb_path, A)
    P = ['http://yago-knowledge.org/resource/ʻElisiva_Fusipala_Taukiʻonetuku']
    N = ['http://yago-knowledge.org/resource/ʻEtuate_Lavulavu']
    P = ['http://yago-knowledge.org/resource/ʻAnaseini_Takipō']
    N = ['http://yago-knowledge.org/resource/ʻAkosita_Lavulavu']
    quality, c = run_evo(kb_path,P,N, timeout=60)
    print(f'Evolearner Accuracy: {quality}')
    print(f'Evolearner concept: {c}')
    f = FittingALC(A, 16, [0],[2], op=frozenset([OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG, OP.LE, OP.GE, OP.DGEQ]),workers=8, max_q=5)
    a, k, sol = f.solve_incr(16)
    print(f'ALCSAT: {a}')
    print(f'ALCSAT {sol.to_dl_concept()}')

def test_clustering():
    A = structure_from_owl(sys.argv[1])
    P, N = read_examples(sys.argv[2])
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A, 16, P,N, op=frozenset([OP.ALL, OP.EX, OP.OR, OP.AND, OP.NEG, OP.LE, OP.GE, OP.DGEQ]),workers=8, max_q=5, clustering = 1)
    a, k, sol = f.solve_incr_approx(7)
    
def tmp(dir_path):
    k = 0
    for d in filter(lambda s : not s.startswith(".") and os.path.isdir(os.path.join(dir_path,s)) ,os.listdir(dir_path)):   
        json_path = os.path.join(dir_path, d, "results.json")
        jd = dict()
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                jd = json.load(f)
        if jd["TDL"]["concept"] == "None":
            k +=1
    print(k)

def combine_bisim_examples(kb_path,dir_path, dest_dir):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    dirs = sorted(filter(lambda s : not s.startswith(".") and os.path.isdir(os.path.join(dir_path,s)) ,os.listdir(dir_path)),reverse = True)
    for i in range(len(dirs)-1):
        P1,N1 = read_examples(os.path.join(dir_path,dirs[i]))
        P2,N2 = read_examples(os.path.join(dir_path,dirs[i+1]))
        P = P1 + P2
        N = N1 + N2
        d_dir = os.path.join(dest_dir, f"{len(P)+len(N)}ex-{i!s}" )
        os.mkdir(d_dir)
        write_examples(P,N, os.path.join(d_dir))
        A = structure_from_owl(kb_path)
        ind_map_inv = {v: k for k, v in A.indmap.items()}
        #P_s = [ind_map_inv[x] for x in P]
        #N_s = [ind_map_inv[x] for x in N]
        reduce_size_by_examples2(
            A, P, N, 16, dest=os.path.join(dest_dir, d_dir,"kb_reduced.owl")
        )

def combine_bisim_examples2(kb_path,dir_path, dest_dir, max_per_size = 5):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    dirs = list(filter(lambda s : not s.startswith(".") and os.path.isdir(os.path.join(dir_path,s)) ,os.listdir(dir_path)))
    d = defaultdict(int)
    k = 0
    for i in range(len(dirs)):
        for j in range(i+1,len(dirs)):            
                P1,N1 = read_examples(os.path.join(dir_path,dirs[i]))
                P2,N2 = read_examples(os.path.join(dir_path,dirs[j]))
                P = P1 + P2
                N = N1 + N2
                if d[len(P)+len(N)] < max_per_size:
                    d[len(P)+len(N)] += 1
                    d_dir = os.path.join(dest_dir, f"{len(P)+len(N)}ex-{k!s}" )
                    os.mkdir(d_dir)
                    write_examples(P,N, os.path.join(d_dir))
                    A = structure_from_owl(kb_path)                    
                    reduce_size_by_examples2(
                        A, P, N, 32, dest=os.path.join(dest_dir, d_dir,"kb_reduced.owl")
                    )
                    k+=1


def main():
    #dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alcq_bisim_combined")
    examples_from_bisim_run(sys.argv[1], overwrite_all=False)
    alcq_benchmarks_to_csv(sys.argv[1])    

if __name__ == "__main__":
    main()
