This repository contains ALCSAT, an implementation of bounded fitting for the
description logic ALCQI(f) as described in the paper

> Funk, M., Jung, J. C., & Voellmer, T. (2026). Bounded fitting for expressive description logics. In Proceedings of IJCAI. 

ALCSAT takes as input an OWL knowledge base and examples labelled as positive or
negative. ALCSAT then searches for a description logic concept that covers all
positive examples, excludes all negative examples and is of minimal size.

By default, ALCSAT searches for a concept of the expressive description logic ALCQI(f) that includes
- all concept constructors of ALC,
- number restrictions (Q),
- inverse roles (I), and
- feature values (f).
However, the search can be restricted to any syntactic fragment, e.g. ELI or ALCQ.

ALCSAT shares some implementation details with SPELL
<https://github.com/spell-system/SPELL>, an implementation of bounded fitting
for the description logic EL.

The code in this repository is licensed under the MIT License, see LICENSE.

If you encounter any problem or have a question, feel free to open an issue.

## Requirements
- python >=3.12
- uv package manager: https://docs.astral.sh/uv/ 
- all other dependencies are specified in `pyproject.toml` and are automatically handled by uv.

## Run
ALCSAT can be run from the command-line using `alcsat_cli.py`. The basic usage is

`uv run alcsat_cli.py kb.owl pos.txt neg.txt`

where `kb.owl` is an OWL knowledge base in RDF/XML format and
`pos.txt`/`neg.txt` are text files in which each line is a positive/negative
example from `kb.owl`.

Try it with an example that is included in this repository:

`uv run alcsat_cli.py tests/father.owl tests/father-example/P.txt tests/father-example/N.txt`

ALCSAT should quickly find a concept that covers all positive examples and excludes all negative examples:
```
Satisfiable for k=4, n=6, acc=1.000000, f1=1.000000
AND
 +-- http://example.com/father#male
 +-- EX.http://example.com/father#hasChild
     +-- TOP
```

The `--language` option can be used to choose a syntactic fragment of ALCQI(f). The important ones are
- `el`: exists, and
- `eli`: exists, and, inverse
- `alc`: forall, exists, and, or, neg
- `alci`: forall, exists, and, or, neg, inverse
- `alcq`: number restrictions, forall, exists, and, or, neg    
    - for alcq, the option `--max_q` becomes available to set the maximum values in number restrictions (defaults to 2)
- `alcqif`: everything
but others are available.

The `--mode` options allows switching between exact mode and approximate mode:
- `approx`: (default) search for best approximate fitting that may not cover all positive examples and exclude all negative examples.
- `exact`: only consider exact fittings: concepts that cover all positive examples and exclude all negative examples. This might be faster in certain scenarios.

The `--workers` option can be used to set the number of worker processes (defaults to 1).

## IJCAI2026 Benchmark Reproduction
Instructions to reproduce experimental results reported in our paper _Bounded
Fitting for Expressive Description Logics_ accepted at IJCAI-ECAI 2026 can be
found in the following repository:
https://github.com/SAT-based-Concept-Learning/ALCSAT-IJCAI-reproduce

## ISWC2025 Benchmark Reproduction
Results shown in our paper _Bounded Fitting for the Description Logic ALC_
accepted at ISWC 2025 can be reproduced as follows. Instructions to reproduce
the family benchmarks are in the folder alc_benchmarks in a separate README
file. Instructions and required files to reproduce the results on the SML
benchmarks can be found in the following repository.
https://github.com/SAT-based-Concept-Learning/ALC-SAT-eval