from seizurecast.models.par import LABEL_BKG, LABEL_SEZ

i_ceil = lambda v, lst: next((i for i, x in enumerate(lst) if x > v), None)
"""return the first index where list[index] > v, return None if not found."""
# assert intvs_ is asccending


def post_sezure_s(timestamps, upper_bounds, labels, max_sec=99999, label_bckg=LABEL_BKG, label_sez=LABEL_SEZ):
    """time passed since last seizure

    Args:
        timestamps: list of timestamps in seconds
        upper_bounds: upper bounds of intervals corresponding to labels.
        labels: labels of seizure status
    """
    post = max_sec
    res = []

    if len(timestamps) < 2:
        return ([post])

    assert timestamps[-1] > timestamps[0], 'Timestamps Must be asccending'

    for sec in timestamps:
        iloc = i_ceil(sec, upper_bounds)
        if sec < 0 or iloc is None:
            res.append(None)
            continue
        l = labels[iloc]
        if l == label_bckg:
            post += 1
            post = min(post, max_sec)
        elif l == label_sez:
            post = 0
        else:
            raise Exception(f'Label: {l} Not Recognized!')
        res.append(post)
    return res


def pres_seizure_s(timestamps, upper_bounds, labels, max_sec=99999, label_bckg=LABEL_BKG, label_sez=LABEL_SEZ):
    """time to go before next seizure

    Args:
        timestamps: list of timestamps in seconds
        upper_bounds: upper bounds of intervals corresponding to labels.
        labels: labels of seizure status
    """
    rt = [upper_bounds[-1] - x for x in reversed(timestamps)]
    rit = [upper_bounds[-1] - x for x in reversed([0] + list(upper_bounds)[:-1])]
    res = post_sezure_s(rt, rit, list(reversed(labels)), max_sec=max_sec, label_bckg=label_bckg, label_sez=label_sez)
    return list(reversed(res))
