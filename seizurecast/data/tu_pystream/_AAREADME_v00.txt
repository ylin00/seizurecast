file: _AAREADME.txt

This directory contains a script, nedc_pystream, that demonstrates the
proper way to read an EDF file and access samples of the signal.
This script takes an EDF file as input and streams the samples
to stdout. A paramter file is supported that allows users to 
control which channels are accessed, and what type of montage can
be imposed.

To learn more about the data, please review this document:

 Ferrell, S., Mathew, V., Ahsan, T., & Picone, J. (2019). The Temple
 University Hospital EEG Corpus: Electrode Location and Channel
 abels. Philadelphia, Pennsylvania, USA.
 https://www.isip.piconepress.com/publications/reports/2019/tuh_eeg/electrodes/

Included in this distribution is a simple EDF file, example.edf. This file
contains the following information:

---
nedc_000_[1]: more example.edf 
0       00008836 M 01-JAN-1952 00008836 Age:60                                  
        Startdate 30-APR-2012 00008836_s003 00 X                                
        30.04.1208.44.117936    EDF                                         5   
    1.00000030  EEG FP1-REF     EEG FP2-REF     EEG F3-REF      EEG F4-REF      
EEG C3-REF      EEG C4-REF      EEG P3-REF      EEG P4-REF      EEG O1-REF      
EEG O2-REF      EEG F7-REF      EEG F8-REF      EEG T3-REF      EEG T4-REF      
EEG T5-REF      EEG T6-REF      EEG A1-REF      EEG A2-REF      EEG FZ-REF      
EEG CZ-REF      EEG PZ-REF      EEG ROC-REF     EEG LOC-REF     EEG EKG1-REF    
EEG T1-REF      EEG T2-REF      PHOTIC-REF      IBI             BURSTS          
SUPPR           Unknown                                                         
                Unknown                     
...
---

Notice that the first three channels are "EEG FP1-REF", "EEG FP2-REF",
"EEG F3-REF", and there is an additional channel "EEG T3-REF". We will focus
on these channels in the example below.

The use of this program is best illustrated with four examples:

===============================================================================
(0) Selecting a single channel using partial matching mode:

EEG labels can be unpredictable. For example, the channel "EEG FP1-REF" could
just be labeled "FP1-REF" at another institution. Therefore, we support
two matching modes to identify channels: exact and partial.

The parameter file param_00.txt simply selects the first channel that matches
"FP1-REF" and prints its values to stdout:

 Input:
  ./nedc_pystream.py -p params_00.txt example.edf

 Output:
  no. output channels =   1
  sample no.    0: 
   channel   0 (     EEGFP1-REF,     250.0000):    4497.9858
  sample no.    1: 
   channel   0 (     EEGFP1-REF,     250.0000):    4490.8142
  sample no.    2: 
   channel   0 (     EEGFP1-REF,     250.0000):    4485.0159
  sample no.    3: 
   channel   0 (     EEGFP1-REF,     250.0000):    4479.6753
  sample no.    4: 
   channel   0 (     EEGFP1-REF,     250.0000):    4472.8088
  ...

This is done by the lines:

 MONTAGE {
  channel_selection = FP1
  match_mode = partial
  montage = (null)
 }

This could also be done by setting match_mode to "exact" and setting
montage to "EEG FP1-REF".

===============================================================================
(1) Selecting a range of channels:

Parameter file param_01.txt selects the following channels: EEG FP1-REF, 
EEG FP2-REF, EEG F3-REF, and EEG T3-REF. It does this by setting match_mode
to exact and montage 0, 1, 2 and 3 to their respective values:

 Input:
  ./nedc_pystream.py -p params_01.txt example.edf

 Output:
  no. output channels =   4
  sample no.    0: 
   channel   0 (     EEGFP1-REF,     250.0000):    4497.9858
   channel   1 (     EEGFP2-REF,     250.0000):    4435.5774
   channel   2 (      EEGF3-REF,     250.0000):    4436.0352
   channel   3 (      EEGT3-REF,     250.0000):    4414.6729
  sample no.    1: 
   channel   0 (     EEGFP1-REF,     250.0000):    4490.8142
   channel   1 (     EEGFP2-REF,     250.0000):    4430.2368
   channel   2 (      EEGF3-REF,     250.0000):    4430.8472
   channel   3 (      EEGT3-REF,     250.0000):    4409.0271
  sample no.    2: 
   channel   0 (     EEGFP1-REF,     250.0000):    4485.0159
   channel   1 (     EEGFP2-REF,     250.0000):    4423.9808
   channel   2 (      EEGF3-REF,     250.0000):    4424.5911
   channel   3 (      EEGT3-REF,     250.0000):    4403.0762
  sample no.    3: 
   channel   0 (     EEGFP1-REF,     250.0000):    4479.6753
   channel   1 (     EEGFP2-REF,     250.0000):    4418.3350
   channel   2 (      EEGF3-REF,     250.0000):    4418.7928
   channel   3 (      EEGT3-REF,     250.0000):    4397.2779
  sample no.    4: 
   channel   0 (     EEGFP1-REF,     250.0000):    4472.8088
   channel   1 (     EEGFP2-REF,     250.0000):    4412.3841
   channel   2 (      EEGF3-REF,     250.0000):    4412.9944
   channel   3 (      EEGT3-REF,     250.0000):    4391.4796
  ...
  
This is accomplished by the following lines in the parameter file:
  
 MONTAGE {
  channel_selection = "FP1", "FP2", "F3", "T3"
  match_mode = partial
  montage = (null)
 }

There is a lot of flexibility in the way channels can be selected.

===============================================================================
(2) Implementing a simple montage that renames channels:

Parameter file param_02.txt implements a simple montage. It differences
channels EEG FP1-REF and EEG FP2-REF. 

 Input:
  ./nedc_pystream.py -p params_02.txt example.edf

 Output:
  no. output channels =   4
  sample no.    0: 
   channel   0 (            Joe,     250.0000):    4414.6729
   channel   1 (           Mary,     250.0000):    4436.0352
   channel   2 (           John,     250.0000):    4435.5774
   channel   3 (           Jane,     250.0000):    4497.9858
  sample no.    1: 
   channel   0 (            Joe,     250.0000):    4409.0271
   channel   1 (           Mary,     250.0000):    4430.8472
   channel   2 (           John,     250.0000):    4430.2368
   channel   3 (           Jane,     250.0000):    4490.8142
  sample no.    2: 
   channel   0 (            Joe,     250.0000):    4403.0762
   channel   1 (           Mary,     250.0000):    4424.5911
   channel   2 (           John,     250.0000):    4423.9808
   channel   3 (           Jane,     250.0000):    4485.0159
  sample no.    3: 
   channel   0 (            Joe,     250.0000):    4397.2779
   channel   1 (           Mary,     250.0000):    4418.7928
   channel   2 (           John,     250.0000):    4418.3350
   channel   3 (           Jane,     250.0000):    4479.6753
  sample no.    4: 
   channel   0 (            Joe,     250.0000):    4391.4796
   channel   1 (           Mary,     250.0000):    4412.9944
   channel   2 (           John,     250.0000):    4412.3841
   channel   3 (           Jane,     250.0000):    4472.8088
  ...

This is accomplished by the following lines in the parameter file:

 MONTAGE {
  channel_selection = "FP1", "FP2", "F3", "T3"
  match_mode = partial
  montage =  0, Joe: T3
  montage =  1, Mary: F3
  montage =  2, John: FP2
  montage =  3, Jane: FP1
 }

The line "montage = 0" includes a specification that renames the input
channel labeled "T3" to the output channel labeled "Joe".

===============================================================================
(3) Implementing a simple montage:

Parameter file param_03.txt implements a simple montage. It differences
channels EEG FP1-REF and EEG FP2-REF. 

 Input:
  ./nedc_pystream.py -p params_03.txt example.edf

 Output:
  no. output channels =   1
  sample no.    0: 
   channel   0 (        FP1-FP2,     250.0000):      62.4084
  sample no.    1: 
   channel   0 (        FP1-FP2,     250.0000):      60.5773
  sample no.    2: 
   channel   0 (        FP1-FP2,     250.0000):      61.0351
  sample no.    3: 
   channel   0 (        FP1-FP2,     250.0000):      61.3403
  sample no.    4: 
   channel   0 (        FP1-FP2,     250.0000):      60.4248
  ...

This is accomplished by the following lines in the parameter file:

 MONTAGE {
  channel_selection = "FP1", "FP2"
  match_mode = partial
  montage = 0, FP1-FP2: EEG FP1-REF -- EEG FP2-REF
 }

The line "montage = 0" includes a specification that indicates these
two channels should be differenced. Note that the label for the output
channel is "FP1-FP2".

===============================================================================
(4) Implementing a Temporal Central Parasagittal (TCP) bipolar montage:

Parameter file param_03.txt implements the montage that we use for most
of our research. It produces 21 channels and implements a very specific
differencing of these channels.

 Input:
  ./nedc_pystream.py -p params_04.txt example.edf

 Output:
  no. output channels =  22
  sample no.    0: 
   channel   0 (         FP1-F7,     250.0000):   -4359.8939
   channel   1 (          F7-T3,     250.0000):       7.1716
   channel   2 (          T3-T5,     250.0000):   -4430.5420
   channel   3 (          T5-O1,     250.0000):       1.0681
   channel   4 (         FP2-F8,     250.0000):   -4406.8909
   channel   5 (          F8-T4,     250.0000):      -3.6621
   channel   6 (          T4-T6,     250.0000):   -4418.9454
   channel   7 (          T6-O2,     250.0000):     -54.3212
   channel   8 (          A1-T3,     250.0000):    4426.2696
   channel   9 (          T3-C3,     250.0000):   -4430.5420
   channel  10 (          C3-CZ,     250.0000):   -4478.6072
   channel  11 (          CZ-C4,     250.0000):      38.4521
   channel  12 (          C4-T4,     250.0000):      16.7847
   channel  13 (          T4-A2,     250.0000):   -4418.9454
   channel  14 (         FP1-F3,     250.0000):   -4359.8939
   channel  15 (          F3-C3,     250.0000):    4482.7270
   channel  16 (          C3-P3,     250.0000):   -4478.6072
   channel  17 (          P3-O1,     250.0000):       9.3079
   channel  18 (         FP2-F4,     250.0000):   -4406.8909
   channel  19 (          F4-C4,     250.0000):      -4.4250
   channel  20 (          C4-P4,     250.0000):      16.7847
   channel  21 (          P4-O2,     250.0000):     -57.8308
  sample no.    1: 
   channel   0 (         FP1-F7,     250.0000):   -4356.2318
   channel   1 (          F7-T3,     250.0000):       7.1716
   channel   2 (          T3-T5,     250.0000):   -4424.8963
   channel   3 (          T5-O1,     250.0000):       1.0681
   channel   4 (         FP2-F8,     250.0000):   -4400.9400
   channel   5 (          F8-T4,     250.0000):      -3.9673
   channel   6 (          T4-T6,     250.0000):   -4413.1470
   channel   7 (          T6-O2,     250.0000):     -53.4057
   channel   8 (          A1-T3,     250.0000):    4420.6238
   channel   9 (          T3-C3,     250.0000):   -4424.8963
   channel  10 (          C3-CZ,     250.0000):   -4472.6563
   channel  11 (          CZ-C4,     250.0000):      37.8418
   channel  12 (          C4-T4,     250.0000):      16.7847
   channel  13 (          T4-A2,     250.0000):   -4413.1470
   channel  14 (         FP1-F3,     250.0000):   -4356.2318
   channel  15 (          F3-C3,     250.0000):    4477.0813
   channel  16 (          C3-P3,     250.0000):   -4472.6563
   channel  17 (          P3-O1,     250.0000):       9.4604
   channel  18 (         FP2-F4,     250.0000):   -4400.9400
   channel  19 (          F4-C4,     250.0000):      -4.2725
   channel  20 (          C4-P4,     250.0000):      16.7847
   channel  21 (          P4-O2,     250.0000):     -56.9152
  sample no.    2: 
   channel   0 (         FP1-F7,     250.0000):   -4349.8231
   channel   1 (          F7-T3,     250.0000):       7.1716
   channel   2 (          T3-T5,     250.0000):   -4418.9454
   channel   3 (          T5-O1,     250.0000):       1.2207
   channel   4 (         FP2-F8,     250.0000):   -4395.2943
   channel   5 (          F8-T4,     250.0000):      -3.6621
   channel   6 (          T4-T6,     250.0000):   -4407.1961
   channel   7 (          T6-O2,     250.0000):     -53.8635
   channel   8 (          A1-T3,     250.0000):    4414.6729
   channel   9 (          T3-C3,     250.0000):   -4418.9454
   channel  10 (          C3-CZ,     250.0000):   -4466.5527
   channel  11 (          CZ-C4,     250.0000):      37.9944
   channel  12 (          C4-T4,     250.0000):      16.6321
   channel  13 (          T4-A2,     250.0000):   -4407.1961
   channel  14 (         FP1-F3,     250.0000):   -4349.8231
   channel  15 (          F3-C3,     250.0000):    4470.8252
   channel  16 (          C3-P3,     250.0000):   -4466.5527
   channel  17 (          P3-O1,     250.0000):       9.4604
   channel  18 (         FP2-F4,     250.0000):   -4395.2943
   channel  19 (          F4-C4,     250.0000):      -4.2725
   channel  20 (          C4-P4,     250.0000):      16.6321
   channel  21 (          P4-O2,     250.0000):     -57.2204
  sample no.    3: 
   channel   0 (         FP1-F7,     250.0000):   -4343.5670
   channel   1 (          F7-T3,     250.0000):       7.1716
   channel   2 (          T3-T5,     250.0000):   -4413.1470
   channel   3 (          T5-O1,     250.0000):       1.2207
   channel   4 (         FP2-F8,     250.0000):   -4389.1907
   channel   5 (          F8-T4,     250.0000):      -3.9673
   channel   6 (          T4-T6,     250.0000):   -4401.0926
   channel   7 (          T6-O2,     250.0000):     -53.7109
   channel   8 (          A1-T3,     250.0000):    4408.8746
   channel   9 (          T3-C3,     250.0000):   -4413.1470
   channel  10 (          C3-CZ,     250.0000):   -4460.7544
   channel  11 (          CZ-C4,     250.0000):      37.9944
   channel  12 (          C4-T4,     250.0000):      16.4795
   channel  13 (          T4-A2,     250.0000):   -4401.0926
   channel  14 (         FP1-F3,     250.0000):   -4343.5670
   channel  15 (          F3-C3,     250.0000):    4465.0269
   channel  16 (          C3-P3,     250.0000):   -4460.7544
   channel  17 (          P3-O1,     250.0000):       9.4604
   channel  18 (         FP2-F4,     250.0000):   -4389.1907
   channel  19 (          F4-C4,     250.0000):      -4.1199
   channel  20 (          C4-P4,     250.0000):      16.4795
  ...
  
This is accomplished by the following lines in the parameter file:
  
 MONTAGE {
  channel_selection = (null)
  match_mode = exact
  montage =  0, FP1-F7: EEG FP1-REF --  EEG F7-REF
  montage =  1, F7-T3:  EEG F7-REF  --  EEG T3-REF
  montage =  2, T3-T5:  EEG T3-REF  --  EEG T5-REF
 ...

Note that each output channel is assigned a label (e.g., "FP1-F7").

===============================================================================
Questions about the use of this software should be directed 
to help@nedcdata.org.

Enjoy,

The Neural Engineering Data Consortium
