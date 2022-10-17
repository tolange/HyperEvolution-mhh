from functools import partial


def rosenbrock_function(parameter_dict: dict, a: int =1, b:int =100, **kwargs):
    ''' The Rosenbrock function

    Parameters:
    -----------
    parameter_dict : dict
        Dictionary containing the hyperparameters (the coordinates of the
        point to be evaluated)
    [a=1] : float
        The parameter 'a' of the Rosenbrock function
    [b=100] : float
        The parameter 'b' of the Roisenbrock function

    Returns:
    -------
    score : float
        The function valueat the coordinates 'x' and 'y'. Returns the negative
        Rosenbrock function value.
    '''
    score = (
        (a - parameter_dict['x'])**2
        + b*(parameter_dict['y']- parameter_dict['x']**2)**2
    )
    return score


def ensemble_rosenbrock(
        parameter_dicts: list,
        true_values: dict ={'a': 1, 'b': 100}
):
    ''' Calcualtes the Rosenbrock function value for the ensemble. The function
    'partial' is used in order to pass our kwargs to the map.

    Parameters:
    -----------
    parameter_dicts : list of dicts
        List of the coordinate dictionaries
    true_values : dict
        Dummy

    Returns:
    --------
    scores : list
        Scores for each member in the ensemble
    '''
    mapfunc = partial(rosenbrock_function, **true_values)
    return list(map(mapfunc, parameter_dicts))
