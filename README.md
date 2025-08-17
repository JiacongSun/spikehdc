# spikehdc

This repository is for testing HDC on spike sorting applications.

## Prerequisite

To get started, you can install all packages directly through pip using the pip-requirements.txt with the command:

```
$ pip install -r requirements.txt
```

## Getting started

To run the test, enter the following command:

```
$ python expr.py
```

There are following parameters:

*regenerate_hypervector*: [bool] generate hypervectors for signal levels (IM) and timing (EM).

*fs*: [float] signal sampling frequency (Hz).

*training_time*: [float] how long the signal clip is to used for training

*training_window_length*: [int] how many samples in one window.

*testcase_folder*: [str] the folder of input signal files.

*testcase_list*: [list[str]] a list of training testcases.

## To plot the signals

`Quiroga_read.py` can be used for plotting signals. It has the following parameters:

*time_to_plot*: [float] how long the time to plot in the png.
