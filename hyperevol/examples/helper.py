import os
import json


def read_cfg(path: str) -> dict:
    ''' Reads the .json file for the settings

    Args:
        path : str
            Path to the configuration file

    Returns:
        cfg : dict
            Configuration read from the given path
    '''
    with open(path, 'rt') as cfg_file:
        cfg = json.load(cfg_file)
    return cfg


def save_results(parameters: dict, fitness: float, output_dir: str) -> None:
    ''' Saves the results to two separate files: parameters to
    "optimal_parameters.json" and fitness score to "fitness.json"

    Args:
        parameters : dict
            The best found parameters to be saved
        fitness : float
            The fitness corresponding to the best found parameters
        output_dir : str
            The directory where the output will be written

    Returns:
        None
    '''
    param_out_path = os.path.join(output_dir, 'optimal_parameters.json')
    fitness_out_path = os.path.join(output_dir, 'fitness.json')
    with open(param_out_path, 'wt') as param_file:
        json.dump(parameters, param_file, indent=4)
    with open(fitness_out_path, 'wt') as fitness_file:
        json.dump(fitness, fitness_file, indent=4)
    print(f"Results saved to:\n\t{param_out_path} \n and\n\t{fitness_out_path}")