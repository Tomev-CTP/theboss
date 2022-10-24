__author__ = "Tomasz Rybotycki"

"""
    This script contains some math methods that are the most general, but are also
    useful in the BS.
"""


from typing import Sequence
from numpy.random import randint, random


def choice(values: Sequence[int], weights: Sequence[float] = None) -> int:
    """
    Returns one of the values according to specified weights. If weights aren't
    specified properly, the method samples value uniformly at random.

    Notice that in this scenario we only want to get the number of particles left after
    application of uniform losses, hence the values are of type int and weights of
    type float.

    We implement our version of choice, as it seems that numpy.random.choice is very
    slow.

    :param values:
        Values to sample from.

    :param weights:
        Weights according to which the sampling will be performed.

    :return:
        Sampled value.
    """
    if weights is None:
        weights = list()

    if len(values) != len(weights):
        return values[randint(0, len(values))]

    weights_sum: float = 0
    random_number: float = random()

    for i in range(len(values)):
        weights_sum += weights[i]
        if weights_sum > random_number:
            return values[i]
