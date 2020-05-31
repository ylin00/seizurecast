import tu_pystream.nedc_pystream as ps


def read_1_token(token_path):
    """Read EDF file and apply its corresponding montage info

    Args:
        edf_filepath: file path to the edf file. Should follow the same
            nameing rules defined in
            https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg/_DOCS/conventions/filenames_v00.txt

    Returns:
        standardized EDF format TBD

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


def get_pre_sez(time, bkg_intvs, sez_intvs):
    """ Compute pre_sez labels from background and seizure intervals
    Args:
        time: time array
        bkg_intvs: list of list of background intervals
        sez_intvs: list of list of sezuire intervals

    Returns:
        array of same length of TIME, with labels of [0, 1, 2, 3, 4]
        representing ['bckg', 'pres', 'seiz', 'post', NaN]
    """
    raise NotImplementedError


def plot_eeg(dataframe, tmin, tmax, fsamp):
    dataframe[slice(int(tmin*fsamp), int(tmax*fsamp))].plot(legend=False)


if __name__ == '__main__':
    pass