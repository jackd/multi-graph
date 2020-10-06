# multi-graph

Package for simultaneously building and connecting multiple tensorflow graphs for data pipelining.

## Quick Start

```bash
pip install tensorflow>=2  # could be tf-nightly
git clone https://github.com/jackd/multi-graph.git
pip install -e multi-graph
pip install absl-py tensorflow-datasets  # for example below
python multi-graph/examples/mnist.py
```

To add as a requirement for an external package:

```txt
git+git://github.com/jackd/multi-graph.git
```

## Usage

See [examples/mnist.py](examples/mnist.py)

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
