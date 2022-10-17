# Examples

## Rosenbrock function

### Rosenbrock function minimization using particle swarm optimization

To run the example, where particle swarm is optimizing the Rosenbrock function:

```bash
python rb_w_pso.py -o <output_dir>
```

### Rosenbrock function minimization using bayesian optimization optimization

To run the example, where bayesian optimization is used for optimizing the Rosenbrock function:

```bash
python rb_w_bo.py -o <output_dir>
```

## Machine learning

The only thing one needs to specify for a given optimizer is the scoring function. A specific example is not (yet) written for a ML workflow here. However the general structure that works well is presented here for writing the scoring function:

1) Wrapper that takes a list of hyperparameter sets as an input and evaluates each parameter set.

    1.1) Evaluations can be local, so launched sequentially. However since ML evaluation takes a long time, then this is not a good solution. For this one would write for example a script that creates a shell script, which is called in order to submit a job to the queue of the cluster. Depending on the submission queue, the process will be most probably different.

    1.2) The same wrapper waits until all the jobs are done and collects the outputs.

2) Returns the fitnesses in the correct order.