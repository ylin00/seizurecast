import os
from src.data import tu_pystream as ps
import numpy.testing as testing


def test_nedc_load_parameters_lbl():
    train_path = '../tusz_1_5_2/edf/train'
    tcp_type = '01_tcp_ar'
    patient_group = '023'
    patient = '00002348'
    session = 's008_2015_07_21'
    token = '00002348_s008_t000'
    token_path = os.path.join(train_path, tcp_type, patient_group, patient,
                              session, token)

    """Load parameters"""
    ### load parameters
    params1 = ps.nedc_load_parameters('tu_pystream/params_04.txt')
    params2 = ps.nedc_load_parameters_lbl(token_path + '.lbl')

    testing.assert_array_equal(params1.keys(), params2.keys())
    testing.assert_array_equal(params1['montage'], params2['montage'])
    testing.assert_array_equal(params1['channel_selection'], params2['channel_selection'])
    testing.assert_array_equal(params1['match_mode'], params2['match_mode'])


