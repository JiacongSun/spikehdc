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


## TODOs

- The test dataset is old. To replace with new ones (multi-channel) and retest.

- The classical frontend is already with low power. HDC can be combined with the classical frontend, which detects spikes first to save power for HDC.

- Look for some references works, check their accuracy and hardware performance.

- Classical algorithms can achieve >97% accuracy, but HDC can only get >80%. The accuracy gap is to be fixed.

## TODOs

- ✅ Apply circular shift for timing.
- ✅ Apply Item Memory on signal level.
- ✅ Apply the frontend detection.
- ❎ Apply sliding windows.
- ☐ Add Neo-DVT frontend.
- ☐ Fix accuracy calculation function.
- ☐ Add Delft dataset.
- ☐ Add runtime training.
- ☐ Add standard calculation for threshold (OR Delft: SOM zi zu zhi wang luo).

Classification accuracy recording:

Dataset: Easy1_noise005, training: 0.5s, test: 0.5s-10.5s

baseline: 86.44%
+ circular shift: 89.5%
+ item memory: 94.24%
+ frontend detection: 99.00%

Dataset: Easy2_noise005, training: 0.5s, test: 0.5s-10.5s
baseline: 85.81%
+ circular shift: 90.88%
+ item memory: 94.2%
+ frontend detection: 98.62%


