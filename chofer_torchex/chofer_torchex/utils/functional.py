import torch


def collection_cascade(input_, stop_predicate: callable, function_to_apply: callable):
    if stop_predicate(input_):
        return function_to_apply(input_)
    elif isinstance(input_, list or tuple):
        return [collection_cascade(x,
                                   stop_predicate=stop_predicate,
                                   function_to_apply=function_to_apply) for x in input_]
    elif isinstance(input_, dict):
        return {k: collection_cascade(v,
                                      stop_predicate=stop_predicate,
                                      function_to_apply=function_to_apply) for k, v in input_.items()}
    else:
        raise ValueError('Unknown type collection type. Expected list, tuple, dict but got {}'
                         .format(type(input_)))


def cuda_cascade(input_, **kwargs):
    return collection_cascade(input_,
                              stop_predicate=lambda x: isinstance(x, torch.Tensor),
                              function_to_apply=lambda x: x.cuda(**kwargs))
