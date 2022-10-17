# HyperEvolution
Algorithms for hyperparameter optimization


### Installation

First make sure you are using python3 ```python --version```. To set python3 as
your default, an easy way is just to alias python to python3:

```bash
echo 'alias python="python3"' >> $HOME/.bashrc
echo 'alias pip="pip3"' >> $HOME/.bashrc
source $HOME/.bashrc
```

Clone the repository:

```bash
git clone git@github.com:Laurits7/HyperEvolution.git
cd HyperEvolution
```

Create a virtual environment and activate it:
```bash
python -m venv Hopt
source Hopt/bin/activate
```

And install the package (-e for the editable version):

```bash
pip install -e .
```

Now every time you need to run the code, you only need to source the environment again

### Examples

The example optimizations can be found under [examples](hyperevol/examples)



### Additional notes:

The optimization algorithms presented here are doing function minimization, so in case you want to maximize (for example AUC) you need to return the score by the scoring function with a minus sign.


---

### References:

The work presented here is based on these two papers:

Tani, Laurits, Diana Rand, Christian Veelken, and Mario Kadastik. 2021.
“Evolutionary Algorithms for Hyperparameter Optimization in Machine
Learning for Application in High Energy Physics.” *The European Physical
Journal C* 81 (2): 1–9.

Tani, Laurits, and Christian Veelken. 2022. “Comparison of Bayesian and
Particle Swarm Algorithms for Hyperparameter Optimisation in Machine
Learning Applications in High Energy Physics.” *arXiv Preprint
arXiv:2201.06809*.
