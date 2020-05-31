"""
1. read edf
    . read edf returns raw: a list of lists
    . read .tse_bi, returns label_array: time array with labels.
    . take label_array and raw, returns dataset and label_array : a list of
        data, each block is 1 second, with fixed size. width is number of
        channels in certain standard order.
2. get labels
3. train models
4. validate models
"""
import tu_pystream.nedc_pystream as ps

LABEL_BKG = 'bckg'
LABEL_PRE = 'pres'
LABEL_SEZ = 'seiz'
LABEL_POS = 'post'
LABEL_NAN = ''


def read_1_token(token_path):
    """Read EDF file and apply its corresponding montage info

    Args:
        edf_filepath: file path to the edf file. Should follow the same
            nameing rules defined in
            https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg/_DOCS/conventions/filenames_v00.txt

    Returns:
        fsamp_mont: list of sampling rate
        sig_mont: list of list of signals in micro Volt
        labels_mont: list of labels.

    """

    """Load parameters"""
    params = ps.nedc_load_parameters_lbl(token_path + '.lbl')

    """read edf"""
    fsamp, sig, labels = ps.nedc_load_edf(token_path + '.edf')

    """select channels"""
    fsamp_sel, sig_sel, labels_sel = ps.nedc_select_channels(params, fsamp, sig,
                                                             labels)

    """apply montage"""
    fsamp_mont, sig_mont, labels_mont = ps.nedc_apply_montage(params, fsamp_sel,
                                                              sig_sel,
                                                              labels_sel)
    return fsamp_mont, sig_mont, labels_mont


def read_1_session(session_path):
    """Read multiple tokens of the same session

    Args:
        session_path: file path to a session.
            e.g. tuh_eeg/v1.0.0/edf/00000037/00001234/s001_2012_03_06

    Returns:
        standard data TBD
    """
    pass


def read_1_patient(patient_folder):
    """Read multiple sessions belonging to the same patient

    Args:
        patient_folder: path to a patient folder
            e.g.: tuh_eeg/v1.0.0/edf/00000037

    Returns:
        standard data TBD

    """
    pass


def load_tse_bi(token_path):
    """Reads .tse_bi file and returns backgroud and seizure intervals

    Args:
        token_path: path to token file
            e.g. tuh_eeg/v1.0.0/edf/00000037/00001234/s001_2012_03_06
            /00001234_s001_t000

    Returns:
        intvs: list of intervals
        labels: list of labels

    """
    intvs, labels = [], []
    with open(token_path+'.tse_bi', 'r') as fp:
        for line in fp:
            line = line.replace('\n', '').split(' ')
            if line[0] == 'version':
                if line[2] != 'tse_v1.0.0':
                    print("tse_bi file must be version='tse_v1.0.0'")
                    exit(-1)
                else:
                    continue
            elif line[0] is not '':
                intvs.append([float(line[0]), float(line[1])])
                labels.append(line[2])
            else:
                continue
    return intvs, labels


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
                _intvs.extend([[beg, min(pos, end)],
                               [min(pos, end), max(beg, pre)],
                               [max(beg, pre), max(beg, gap)],
                               [max(beg, gap), end]])
                _labls.extend([LABEL_POS, LABEL_BKG, LABEL_PRE, LABEL_NAN])
        elif lbl == LABEL_SEZ:
            _intvs.append([beg, end])
            _labls.append(LABEL_SEZ)
        else:
            raise ValueError("Illegal LABELS")

    _labls = [_labls[i] for i, intv in enumerate(_intvs) if intv[0] < intv[1]]
    _intvs = [_intvs[i] for i, intv in enumerate(_intvs) if intv[0] < intv[1]]

    return _intvs, _labls


def sort_channel(sig, ch_labels, std_labels):
    """sort channel based on standard labels

    Args:
        sig: array of array of EEG signals.
        ch_labels: array of channel labels. Len(LABELS) must = width of SIG
        std_labels: array of standard channel labels. must of same len as LABELS

    Returns:
        sig_sorted: array of array of EEG signals

    """
    raise NotImplementedError


def chop_signal(sig, fsamp:int):
    """Generate dataset from EEG signals and labels

    Args:
        sig: array of array of EEG signals.
        fsamp: integer, sampling rate.

    Returns:
        data: list of list of list of EEG signals.
            DATA[i] has length of FSAMP[i] and width of len(LABELS)
        flag:
            0 if all DATA[i] has same length.
            1 if otherwise.
    """
    raise NotImplementedError


def get_data_label(sig, fsamp, ch_labels, intvs, labels, opt=None):
    """return data and labels

    returns dataset and label_array : a list of data, each block is 1
    second, with fixed size. width is number of channels in certain standard
    order.

    Args:
        sig: array of array of EEG signals.
        fsamp: integer, sampling rate.
        ch_labels: array of channel labels. Len(LABELS) must = width of SIG
        intvs: list of list of intervals
        labels: list of labels. Must be same len as INTVS
        opt: dictionary of options
            len_pre: length of pre-seizure stage in seconds
            len_post: length of post-seizure stage in seconds
            sec_gap: gap between pre-seizure and seizure in seconds

    Returns:
        tuple: dataset: list of data; labels: list of labels

    """
    raise NotImplementedError


def plot_eeg(dataframe, tmin, tmax, fsamp):
    dataframe[slice(int(tmin*fsamp), int(tmax*fsamp))].plot(legend=False)


if __name__ == '__main__':
    pass