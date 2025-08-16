import logging
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tqdm

def u_gen_rand_hv(D,p=0.5) -> list[int]:
    """! function copied from stef: function to generate a random binary hypervector
    @param D: Dimension of the hypervector
    @param p: Density level of 1s
    @return: List of generated binary hypervector, [1, 0, 1, 0, ...]
    """
    #use long index list, generate range and then permute them
    # and set all numbers < p*D to 1 to get right density
    hv = [*range(D)]
    np.random.shuffle(hv)
    for i in range(len(hv)):
        if hv[i] < p*D:
          hv[i] = 1
        else:
          hv[i] = 0
    return hv

def bundle_dense(block) -> np.ndarray:
  """! Copied from stef: Combine dense blocks by majority voting
  @param block: Input block of hypervectors
  @return: Combined hypervector
  """
  if ((len(block)%2) == 0):
     block = block[0:len(block)-1]
  sums = np.sum(block, axis = 0)
  for x in range(len(sums)):
    if (sums[x] <= (len(block))/2):
      sums[x] = 0
    else:
      sums[x] = 1
  return sums

def generate_hypervector(hv_length: int = 1024, hv_count: int = 401) -> dict:
    """! Generate hypervector for different signal levels
    @param hv_length: Length of the hypervector
    @param hv_count: Number of hypervectors to generate
    @return: Array of generated hypervectors
    """

    logging.info("Generating hypervector...")
    hv_dict = {}
    for i in range(hv_count):
        new_hv = u_gen_rand_hv(hv_length)
        hv_dict[i] = new_hv
    return hv_dict

def run_hdc(regenerate_hypervector: bool = True,
            testcase_folder: str = "../yunzhu/",
            testcase: str = "C_Easy1_noise005.mat",
            hv_length: int = 1024,
            im_hv_count: int = 401,
            em_hv_count: int = 1,
            training_window_length: int = 256,
            fs: int = 24000,
            training_time: float = 0.5) -> any:
    """! Run the HDC process
    @param regenerate_hypervector: Whether to regenerate hypervectors
    @param testcase_folder: Folder containing the test cases
    @param testcase: Name of the test case file
    @param hv_length: Length of the hypervector
    @param im_hv_count: Number of hypervectors for input signal levels
    @param em_hv_count: Number of hypervectors for input channel IDs
    @param training_window_length: Length of the training window
    @param fs: Sampling frequency (Hz)
    @param training_time: Training time (seconds)
    """
    logging.info("Running HDC...")
    # Generate hypervectors if needed
    if regenerate_hypervector:
        IM = generate_hypervector(hv_length=hv_length, hv_count=im_hv_count)
        EM = generate_hypervector(hv_length=hv_length, hv_count=em_hv_count)
        # write to files
        with open(f"{testcase_folder}IM_hv.txt", "w") as f:
            for key, value in IM.items():
                f.write(f"{value}\n")
        with open(f"{testcase_folder}EM_hv.txt", "w") as f:
            for key, value in EM.items():
                f.write(f"{value}\n")
    else:
        # load from files
        IM = {}
        EM = {}
        with open(f"{testcase_folder}IM_hv.txt", "r") as f:
            i = 0
            for line in f:
                IM[i] = list(map(int, line[1: -2].split(",")))
                i += 1
        with open(f"{testcase_folder}EM_hv.txt", "r") as f:
            i = 0
            for line in f:
                EM[i] = list(map(int, line[1: -2].split(",")))
                i += 1

    # load the sample
    matlab_source_data = sio.loadmat(f"{testcase_folder}{testcase}")
    recording_traces = matlab_source_data["data"][0]
    spike_times = matlab_source_data["spike_times"][0][0][0]
    spike_times += 24  # 24 is for label correction
    spike_class = matlab_source_data["spike_class"][0][0][0]

    # assertion: curent IM supports range [-2, -2] with a step of 0.01
    recording_min = np.min(recording_traces)
    recording_max = np.max(recording_traces)
    assert recording_min >= -2.01 and recording_max <= 2.01, "Recording traces are out of bounds."

    # training
    logging.info("Training...")

    training_sequence = recording_traces[0: int(fs * training_time)]
    num_of_training_windows = len(training_sequence) // training_window_length
    training_hv: list = []  # record of hypervectors of different windows
    training_spike_labels: list = []  # record of spike labels for different windows
    training_spike_class: list = []  # record of spike class for different windows

    # generate spike labels for windows
    spike_times = spike_times[spike_times <= int(fs * training_time)]
    window_having_spike = [int(t // training_window_length) for t in spike_times]
    training_spike_labels = [1 if i in window_having_spike else 0 for i in range(num_of_training_windows)]

    # check the minimal diff of spike time
    spike_times_diff = [spike_times[i+1] - spike_times[i] for i in range(len(spike_times) - 1)]
    spike_times_diff_min = np.min(spike_times_diff) if spike_times_diff else 0
    if spike_times_diff_min < training_window_length:
        logging.warning(f"Spike times ({spike_times_diff_min}) are too close to the window length "
                        f"{training_window_length}, may cause issues in training.")

    # generate spike class for windows
    spike_class = spike_class[0: len(spike_times)]
    training_spike_class = [int(spike_class[window_having_spike.index(i)] if i in window_having_spike else 0) for i in range(num_of_training_windows)]

    # training on spike signals
    for i in tqdm.tqdm(range(num_of_training_windows), ascii="░▒█", desc="Training windows"):
        window = training_sequence[i * training_window_length: (i + 1) * training_window_length]
        # process the window
        block_hv: list = []
        for j in range(len(window)):
            signal_hv: list = []
            signal_level = window[j]
            signal_to_im = IM[int((signal_level + 2) * 100)]
            signal_hv.append(np.logical_xor(signal_to_im, EM[0]))  # use EM[0] as there is only one channel
            # compress signal_hv
            compressed_signal_hv = bundle_dense(signal_hv)
            block_hv.append(compressed_signal_hv)
        training_hv.append(bundle_dense(block_hv))

    breakpoint()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # parameters
    regenerate_hypervector = False
    fs = 24000 # sampling frequency (Hz)
    training_time = 0.5 # seconds
    training_window_length = 30 # number of samples
    testcase_folder = "./"
    testcase_list = ["C_Easy1_noise005.mat",
                     "C_Easy2_noise005.mat"]

    for testcase in testcase_list:
        run_hdc(regenerate_hypervector=regenerate_hypervector,
                testcase_folder=testcase_folder,
                testcase=testcase,
                training_window_length=training_window_length,
                fs=fs,
                training_time=training_time)