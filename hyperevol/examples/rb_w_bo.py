''' This script runs the Particle swarm optimization (PSO) with a given batch
size for 1000 repeats each optimization consists out of 10k total evaluations.
Call with 'python'

Usage: rb_w_bo.py --output_dir=DIR

Options:
    -o --output_dir=DIR             Directory of the output
'''
import os
import json
import docopt
import numpy as np
from helper import read_cfg, save_results
from rosenbrock_scoring import ensemble_rosenbrock
from hyperevol.tools import bayesian_optimization as bo


def main(output_dir: str) -> None:
    ''' Runs the particle swarm optimization to optimize the Rosenbrock function
    and saves the result to a file in the specified folder.
        Since no additional parameters need to be given to the Rosenbrock fn,
    then no additional 'settings=xyz' will be specified for PSO here.
        After optimization, the other logging info (e.g. score evolution) can
    be accessed easily by e.g "swarm.global_bests"

    Args:
        output_dir : str
            The directory where the output will be written

    Returns:
        None
    '''
    os.makedirs(output_dir, exist_ok=True)
    bo_cfg = read_cfg('config/bo_cfg.json')
    hyperparameters = read_cfg('config/rosenbrock_cfg.json')
    bo_best_parameters, bo_best_fitness = bo.optimize(
                                                      ensemble_rosenbrock,
                                                      hyperparameters,
                                                      **bo_cfg)
    print(f"Found optimal parameters: {bo_best_parameters}")
    print(f"Found optimal value with optimal parameters: {bo_best_fitness}")
    print("--------------------------------------------------------")
    print("Saving results:")
    save_results(bo_best_parameters, bo_best_fitness, output_dir)


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        output_dir = arguments['--output_dir']
        main(output_dir)
    except docopt.DocoptExit as e:
        print(e)
