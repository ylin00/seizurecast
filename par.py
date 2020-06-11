# relabeling config
LEN_PRE = 180
LEN_POS = 60
SEC_GAP = 60
SAMPLING_RATE = 256
LABEL_BKG = 'bckg'
LABEL_PRE = 'pres'
LABEL_SEZ = 'seiz'
LABEL_POS = 'post'
LABEL_NAN = ''
EVENT_ID = {LABEL_BKG: 0,
            LABEL_PRE: 1,
            LABEL_SEZ: 2,
            LABEL_POS: 3,
            LABEL_NAN: 4}
STD_CHANNEL_01_AR = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4',
                     'T4-T6', 'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4',
                     'C4-T4', 'T4-A2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                     'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']