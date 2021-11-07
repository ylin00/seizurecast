from seizurecast.data.label import post_sezure_s, pres_seizure_s
from seizurecast.data.parameters import LABEL_SEZ, LABEL_BKG
import numpy as np


def test_post_sezure_s():
    timestamp = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 1.1, 1.2]
    upperbounds = [0.3, 0.5, 1.2]
    lbl = [LABEL_BKG, LABEL_SEZ, LABEL_BKG]
    np.testing.assert_array_almost_equal(pres_seizure_s(timestamp, upperbounds, lbl),
                                         [0.3, 0.2, 0.1, 0, 0, 99999, 99999, 99999, 99999, 99999])
    np.testing.assert_array_almost_equal(post_sezure_s(timestamp, upperbounds, lbl),
                                         [99999, 99999, 99999, 99999, 0, 0, 0.1, 0.5, 0.6, 0.7])
