from skopt import Optimizer
import numpy as np


def get_dimensions(hyperparameters: dict) -> list:
    ''' Infers the dimension of the hyperparameter plane

    Args:
        hyperparameters : dict
            The dictionary containing information of the hyperparameters

    Returns:
        dimensions : list
            The size of the hyperparameter plane on each axis.
    '''
    dimensions = []
    for hyperpar in hyperparameters.values():
        min_ = hyperpar['min'] if not hyperpar['exp'] else np.exp(hyperpar['min'])
        max_ = hyperpar['max'] if not hyperpar['exp'] else np.exp(hyperpar['max'])
        dim = (min_, max_)
        dimensions.append(dim)
    return dimensions


def optimize(
            objective_function,
            hyperparameters,
            settings={},
            seed=42,
            n_parallel=5,
            n_iter=30,
            acq_optimizer="lbfgs",
            base_estimator='gp',
            initial_point_generator='lhs',
            acq_func="EI",
            n_initial_points=100,
            ask_strategy='cl_min',
            **kwargs
    ):
    '''
    Acquisition function optimizer:
        l-BFGS or Limited memory Broyden-Fletcher-Goldfarb-Shanno
        See more at arXiv:1807.02811 [stat.ML]
        "A tutorial on Bayesian Optimization"
        l-BFGS takes a lot less time and space than BFGS since it doesnt store
        the full Hessian, only an approximation

    Initial point generator:
        LHS or Latin Hypercube Sequence.
        "Maximum projection designs for computer experiments"
    '''
    dimensions = get_dimensions(hyperparameters)
    opt = Optimizer(
        dimensions,
        base_estimator=base_estimator,
        n_initial_points=n_initial_points,
        initial_point_generator=initial_point_generator,
        n_jobs=1,
        acq_func=acq_func,
        acq_optimizer=acq_optimizer,
        random_state=seed,
    )
    best_hyperparameters = {}
    best_value = 99e99
    suggestions = opt.ask(
        n_points=n_initial_points,
        strategy=ask_strategy
    )
    suggested_values = []
    for suggestion in suggestions:
        suggested_values.append({
            hyperparameter: s for hyperparameter, s in zip(hyperparameters.keys(), suggestion)
        })
    y = objective_function(suggested_values)
    opt.tell(suggestions, y)
    for i in range(n_iter):
        print('Iteration %s' %i)
        start = time.time()
        suggestions = opt.ask(
            n_points=n_parallel,
            strategy=ask_strategy
        )
        suggested_values = []
        for suggestion in suggestions:
            suggested_values.append({
                hyperparameter: s for hyperparameter, s in zip(hyperparameters.keys(), suggestion)
            })
        y = objective_function(suggested_values)
        opt.tell(suggestions, y)
        iter_min = np.argmin(y)
        if y[iter_min] < best_value:
            best_value = y[iter_min]
            best_hyperparameters = suggested_values[iter_min]
    return best_value, best_hyperparameters
