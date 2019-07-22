import torch


def histogram_intersection_loss(input_: torch.Tensor,
                                target: torch.Tensor,
                                size_average: bool = True,
                                reduce: bool = True,
                                symmetric_version: bool = True) -> torch.Tensor:
    r"""
    This loss function is based on the `Histogram Intersection` score introduced in
    #TODO ref needed

    The output is the *negative* Histogram Intersection Score.

    Args:
        input_ (Tensor): :math:`(N, B)` where `N = batch size` and `B = number of classes`
        target (Tensor): :math:`(N, B)` where `N = batch size` and `B = number of classes`
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                :attr:`size_average` is set to ``False``, the losses are instead summed
                for each minibatch. Ignored if :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional):
        symmetric_version (bool, optional): By default, the symmetric version of histogram intersection
                is used. If false the asymmetric version is used. Default: ``True``

    Returns: Tensor.

    """
    assert input_.size() == target.size(), \
        "input.size() != target.size(): {} != {}!".format(input_.size(), target.size())
    assert input_.dim() == target.dim() == 2, \
        "input, target must be 2 dimensional. Got dim {} resp. {}".format(input_.dim(), target.dim())

    minima = input_.min(target)
    summed_minima = minima.sum(dim=1)

    if symmetric_version:
        normalization_factor = (input_.sum(dim=1)).max(target.sum(dim=1))
    else:
        normalization_factor = target.sum(dim=1)

    loss = summed_minima / normalization_factor

    if reduce:
        loss = sum(loss)

        if size_average:
            loss = loss / input_.size(0)

    return -loss
