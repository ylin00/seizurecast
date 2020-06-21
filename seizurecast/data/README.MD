The data submodule hosts functions that make use of tu_pystream package

Examples
--------
See examples.ipynb

. read edf returns raw: a list of lists
. read .tse_bi, returns label_array: time array with labels.
. take label_array and raw, returns dataset and label_array : a list of
    data, each block is 1 second, with fixed size. width is number of
    channels in certain standard order.
