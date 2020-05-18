

def unravel_group_params(unfolded_parameters_group):
    """Take a dict('group:k'-> p) and return a dict(group -> {k->p})
    """
    parameters_group = {}
    for k, p in unfolded_parameters_group.items():
        group_name, k = k.split(':')
        group_params = parameters_group.get(group_name, {})
        group_params[k] = p
        parameters_group[group_name] = group_params
    return parameters_group


def ravel_group_params(parameters_group):
    """Take a dict(group -> {k->p}) and return a dict('group:k'-> p)
    """

    return {f'{group_name}:{k}': p
            for group_name, group_params in parameters_group.items()
            for k, p in group_params.items()}


def list_parameters_from_groups(parameter_groups, groups):
    """Return a list of all the parameters in a list of groups
    """
    return [
        p for group in groups
        for p in parameter_groups[group].values()
    ]
