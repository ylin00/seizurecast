"""
Label manipulation
"""
from seizurecast.data.parameters import LABEL_BKG, LABEL_SEZ, LABEL_PRE, LABEL_NAN, LABEL_POS
from seizurecast.utils import i_ceil


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

    sec_before = 0
    for _, sec in enumerate(timestamps):
        iloc = i_ceil(sec, upper_bounds)
        if sec < 0 or iloc is None:
            res.append(None)
            continue
        l = labels[iloc]
        if l == label_bckg:
            post += (sec - sec_before)
            post = min(post, max_sec)
        elif l == label_sez:
            post = 0
        else:
            raise Exception(f'Label: {l} Not Recognized!')
        res.append(post)
        sec_before = sec
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


def relabel_tse_bi(intvs, labels, len_pre=100, len_post=300, sec_gap=0):
    """ Compute labels from background and seizure intervals

    Args:
        intvs: list of list of intervals
        labels: list of labels of background and seizure. Must be same len as INTVS
        len_pre: length of pre-seizure stage in seconds
        len_post: length of post-seizure stage in seconds
        sec_gap: gap between pre-seizure and seizure in seconds

    Returns:
        tuple: (Array of intervals, Array of labels).
    """
    _intvs, _labls = [], []
    # Find LABEL_SEZ
    # locate pre, gap and pos
    # assign others to LABEL_BKG
    for i, lbl in enumerate(labels):
        beg, end = intvs[i]
        pos = beg + len_post
        pre = end - sec_gap - len_pre
        gap = end - sec_gap
        if lbl == LABEL_BKG:
            if i == 0 and i == len(labels)-1:
                _intvs.append([beg, end])
                _labls.append(LABEL_BKG)
            elif i == 0:
                _intvs.extend([[beg, max(beg, pre)],
                               [max(beg, pre), max(beg, gap)],
                               [max(beg, gap), end]])
                _labls.extend([LABEL_BKG, LABEL_PRE, LABEL_NAN])
            elif i == len(labels)-1:
                _intvs.extend([[beg, min(pos, end)],
                               [min(pos, end), end]])
                _labls.extend([LABEL_POS, LABEL_BKG])
            else:
                pos_end = min(pos, end)
                pre_beg = max(pos_end, pre)
                gap_beg = max(pos_end, gap)
                _intvs.extend([[beg, pos_end],[pos_end, pre_beg],
                               [pre_beg, gap_beg],[gap_beg, end]])
                _labls.extend([LABEL_POS, LABEL_BKG, LABEL_PRE, LABEL_NAN])
        elif lbl == LABEL_SEZ:
            _intvs.append([beg, end])
            _labls.append(LABEL_SEZ)
        else:
            raise ValueError("Illegal LABELS")

    _labls = [_labls[i] for i, intv in enumerate(_intvs) if intv[0] < intv[1]]
    _intvs = [_intvs[i] for i, intv in enumerate(_intvs) if intv[0] < intv[1]]

    return _intvs, _labls