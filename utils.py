import pickle
from pandas import DataFrame


def which_bin(elem, bins):
    """Return the index of first intervals that element is in

    Args:
        elem(float): a number.
        bins: an array of intervals in the format of (lower, upper]

    Returns:
        int: an index of the first interval the element is in. -1 if not found.

    """
    for idx, bounds in enumerate(bins, start=1):
        if bounds[0] < elem <= bounds[1]:
            return idx
    else:
        return -1


def transpose(matrix):
    """Matrix transpose

    Args:
        matrix: list of list

    Returns:
        list: list of list

    """
    return list(map(list, zip(*matrix)))


def save(dataset, labels):
    with open('dataset.pkl', 'wb') as fp:
        pickle.dump(dataset, fp)
    with open('labels.pkl', 'wb') as fp:
        pickle.dump(labels, fp)


def load():
    with open('dataset.pkl', 'rb') as fp:
        d = pickle.load(fp)
    with open('labels.pkl', 'rb') as fp:
        l = pickle.load(fp)
    return d, l


def array3D_to_dataframe(dataset, labels):
    """

    Args:
        dataset: nepochs x nchannels x nsamples
        labels: nepochs x 1

    Returns:
        data.frame

    """
    flat = []

