import logging
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tqdm
import copy
import os
from scipy import signal

def filtering(recording_traces, filter_params,fs):
    filter_en = filter_params['filter_en']
    low_cutoff = filter_params['corner_freq'][0]
    high_cutoff = filter_params['corner_freq'][1]
    order = filter_params['order']  
    axis = filter_params['axis']
    if (filter_en == 1):
        low_cutoff_normalized = low_cutoff / (0.5 * fs)  
        high_cutoff_normalized = high_cutoff / (0.5 * fs)
        b, a = signal.butter(order, [low_cutoff_normalized, high_cutoff_normalized], btype='band', analog=False, output='ba')
        recording_traces = signal.lfilter(b, a, recording_traces,axis = axis)
    else:
        recording_traces = recording_traces
    return recording_traces

def u_gen_rand_hv(dimension, p=0.5) -> list[int]:
    """! function copied from stef: function to generate a random binary hypervector
    @param dimension: Dimension of the hypervector
    @param p: Density level of 1s
    @return: List of generated binary hypervector, [1, 0, 1, 0, ...]
    """
    #use long index list, generate range and then permute them
    # and set all numbers < p*dimension to 1 to get right density
    hv = [*range(dimension)]
    np.random.shuffle(hv)
    for i in range(len(hv)):
        if hv[i] < p*dimension:
            hv[i] = 1
        else:
            hv[i] = 0
    return hv

def bundle_dense(block) -> np.ndarray:
    """! Copied from stef: Combine dense blocks by majority voting
    @param block: Input block of hypervectors
    @return: Combined hypervector
    """
    if((len(block)%2) == 0):
        block = block[0:len(block)-1]
    sums = np.sum(block, axis = 0)
    for x in range(len(sums)):
        if(sums[x] <= (len(block))/2):
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

def generate_item_hypervector(
        hv_length: int = 1024,
        hv_count: int = 401,
        end_distance: float = 1,
        ) -> dict:
    """! Generate hypervector for different signal levels with item memory
    @param hv_length: Length of the hypervector
    @param hv_count: Number of hypervectors to generate
    @param end_distance: End distance for the hypervectors, [0, 1]
    """
    logging.info("Generating item hypervector...")
    hv_dict = {}
    clip_size: float = hv_length * end_distance / hv_count
    new_hv: list = u_gen_rand_hv(hv_length)
    hv_init: np.ndarray = np.array(copy.deepcopy(new_hv), dtype=int)
    for i in range(hv_count):
        hv_dict[i] = new_hv
        clip_start = round(clip_size * i)
        clip_end = round(clip_size * (i + 1))
        clip = new_hv[clip_start: clip_end] if clip_end <= hv_length else new_hv[
            clip_start: hv_length]
        flipped_clip = [0 if bit == 1 else 1 for bit in clip]
        new_hv = np.concatenate(
            (new_hv[:clip_start], flipped_clip, new_hv[clip_end:]))
        new_hv = np.array(new_hv, dtype=int)
        new_hv = new_hv.tolist()
    # logging
    end_hamming_distance = calc_hamming_distance(hv_init, np.array(new_hv))
    logging.info(f"End Hamming distance: "
                 f"{end_hamming_distance} ({end_hamming_distance / hv_length:.2%})")
    return hv_dict

def calc_hamming_distance(hv1: np.ndarray, hv2: np.ndarray) -> int:
    """! Calculate the Hamming distance between two hypervectors
    @param hv1: First hypervector
    @param hv2: Second hypervector
    @return: Hamming distance
    """
    assert np.all((hv1 == 0) | (hv1 == 1)), "hv1 must only contain 0 or 1"
    assert np.all((hv2 == 0) | (hv2 == 1)), "hv2 must only contain 0 or 1"
    # return np.sum(np.array(hv1) != np.array(hv2))
    return int(sum(np.logical_xor(hv1, hv2)))

def load_hypervector(file_path: str) -> dict:
    """! Load hypervectors from a file
    @param file_path: Path to the file containing hypervectors
    @return: Dictionary of hypervectors
    """
    hv_dict = {}
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            hv_dict[i] = list(map(int, line[1: -2].split(",")))
    return hv_dict

def calc_hdc_of_windows(
    training_sequence: list,
    IM: dict,
    EM: dict,
    training_window_length: int,
) -> list:
    """! Calculate HDC for each training window
    @param training_sequence: List of training sequences
    @param IM: Input hypervectors
    @param EM: Encoding hypervectors
    @param training_window_length: Length of the training window
    @return: List of hypervectors for each training window
    """
    num_of_training_windows = len(training_sequence) // training_window_length
    hv_list: list = []  # record of hypervectors of different windows
    for i in tqdm.tqdm(range(num_of_training_windows), ascii="░▒█", desc="Training windows"):
        window = training_sequence[i * training_window_length: (i + 1) * training_window_length]
        # process the window
        block_hv: list = []
        for j in range(len(window)):
            signal_hv: list = []
            signal_level = window[j]
            signal_to_im = IM[int((signal_level + 2) * 100)] # the map is [-2, 2] -> [0, 401]
            signal_hv.append( np.roll(signal_to_im, j).tolist() ) # use circular shift to represent timing
            # signal_hv.append(np.logical_xor(signal_to_im, EM[j])) # use EM to represent timing
            # compress signal_hv
            compressed_signal_hv = bundle_dense(signal_hv)
            block_hv.append(compressed_signal_hv)
        hv_list.append(bundle_dense(block_hv))
    return hv_list

def run_hdc(regenerate_hypervector: bool = True,
            testcase_folder: str = "../yunzhu/",
            testcase: str = "C_Easy1_noise005.mat",
            hv_length: int = 1024,
            im_hv_count: int = 401,
            em_hv_count: int = 100,
            training_window_length: int = 256,
            fs: int = 24000,
            training_time: float = 0.5,
            with_front_end: bool = False,
            filter_params: dict = {'filter_en' : True,'corner_freq':[1000,6000],'order' : 1,'axis' : 0}) -> any:
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
    @param with_front_end: If with front end precisely monitoring spikes
    """
    logging.info("Running HDC...")
    # Generate hypervectors if needed
    if regenerate_hypervector:
        # IM = generate_hypervector(hv_length=hv_length, hv_count=im_hv_count)
        IM = generate_item_hypervector(hv_length=hv_length, hv_count=im_hv_count, end_distance=1)
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
        IM = load_hypervector(f"{testcase_folder}IM_hv.txt")
        EM = load_hypervector(f"{testcase_folder}EM_hv.txt")

    # load the sample
    matlab_source_data = sio.loadmat(f"{testcase_folder}{testcase}")
    recording_traces = matlab_source_data["data"][0]
    recording_traces = filtering(recording_traces, filter_params,fs) #add filtering

    spike_times = matlab_source_data["spike_times"][0][0][0]
    spike_times += 24  # 24 is for label correction
    spike_class = matlab_source_data["spike_class"][0][0][0]

    # assertion: curent IM supports range [-2, -2] with a step of 0.01
    recording_min = np.min(recording_traces)
    recording_max = np.max(recording_traces)
    assert recording_min >= -2.01 and recording_max <= 2.02, \
        f"Recording traces [{recording_min}, {recording_max}] are out of bounds."

    # training
    logging.info("Training...")

    training_spike_labels: list = []  # record of spike labels for different windows
    training_spike_class: list = []  # record of spike class for different windows

    # generate spike labels for windows
    training_sequence = recording_traces[0: int(fs * training_time)]
    num_of_training_windows = len(training_sequence) // training_window_length
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
    if not with_front_end:
        spike_class = np.append(spike_class, 0)  # append 0 for non-spiking window
    class_set = sorted(set(spike_class))
    training_spike_class = [int(spike_class[window_having_spike.index(i)] if i in window_having_spike else 0)
                            for i in range(num_of_training_windows)]

    ######################################################
    ## To be replaced with real frontend processing alg.
    # add frontend processing (only keep spike windows)
    if with_front_end:
        training_sequence_processed = np.array([])
        for tim in spike_times:
            seq_clipped = recording_traces[tim - training_window_length//2: tim + training_window_length//2]
            training_sequence_processed = np.append(training_sequence_processed, seq_clipped)
        training_spike_labels = np.ones(len(training_sequence_processed) // training_window_length)
        training_spike_class = spike_class
        assert len(training_spike_class) == len(training_spike_labels), \
            f"Training spike class and labels must have the same length."
    else:
        training_sequence_processed = training_sequence
        training_spike_labels = training_spike_labels
        training_spike_class = training_spike_class
    ######################################################

    # training on spike signals
    logging.info("S1: Calculate hypervectors per window...")
    training_hv = calc_hdc_of_windows(
        training_sequence=training_sequence_processed,
        IM=IM,
        EM=EM,
        training_window_length=training_window_length
    )

    logging.info("S2: Compress hypervectors per class...")
    # convert to np.array for ease of operations
    training_hv = np.array(training_hv)
    training_spike_labels = np.array(training_spike_labels)
    training_spike_class = np.array(training_spike_class)

    hv_per_class: list = []
    hv_per_class_dict: dict = {}
    zero_in_class = False
    for i in tqdm.tqdm(range(len(class_set)), ascii="░▒█", desc="Compressing hypervectors"):
        class_idx = int(list(class_set)[i])
        all_hv = training_hv[training_spike_class == class_idx]
        compressed_hv = bundle_dense(all_hv)
        hv_per_class.append(compressed_hv)
        hv_per_class_dict[class_idx] = np.array(compressed_hv)
        if class_idx == 0:
            zero_in_class = True

    logging.info("S3: Compress hypervectors for all classes...")
    ## can be used to detect if a window has a spike or not
    if zero_in_class:
        hv_per_class_nz = np.array(hv_per_class[1:])  # remove the zero class
    else:
        hv_per_class_nz = np.array(hv_per_class)
    hv_all_classes: np.array = bundle_dense(hv_per_class_nz)

    # decide the suggested similarity threshold
    # S1: calculate the maximal Hamming distance within each class
    max_hamming_dist_per_class: dict = {}
    average_hamming_dist_per_class: dict = {}
    three_std_hamming_dist_per_class: dict = {}

    for class_idx, hv in hv_per_class_dict.items():
        max_hamming_dist = 0
        average_hamming_dist = []
        all_hv_in_class = training_hv[training_spike_class == class_idx]
        for i in range(len(all_hv_in_class)):
            hamming_dist = calc_hamming_distance(all_hv_in_class[i], hv)
            max_hamming_dist = max(max_hamming_dist, hamming_dist)
            average_hamming_dist.append(hamming_dist)
        max_hamming_dist_per_class[class_idx] = max_hamming_dist
        average_hamming_dist_per_class[class_idx] = int(np.mean(average_hamming_dist))
        three_std_hamming_dist_per_class[class_idx] = np.std(average_hamming_dist) * 3 if average_hamming_dist else 0
    max_hamming_dist = max(max_hamming_dist_per_class.values())
    logging.debug(f"Max Hamming distance per class: {max_hamming_dist_per_class}")
    logging.debug(f"Average Hamming distance per class: {average_hamming_dist_per_class}")
    logging.debug(f"3 Std Hamming distance per class: {three_std_hamming_dist_per_class}")

    # S2: calculate the Hamming distance across class
    min_dist_across_class = float("inf")  # used to check the minimal Hamming distance
    # dict to save Hamming distances from one class to other classes
    hamming_dist_across_class: dict = {}
    class_id_list: list = list(hv_per_class_dict.keys())
    for i in range(len(hv_per_class)):
        hamming_dist_list = []
        all_hv_in_class = training_hv[training_spike_class == class_id_list[i]]
        for j in range(len(hv_per_class)):
            if i == j:
                continue
            hamming_dist_per_hv: list = []
            for k in range(len(all_hv_in_class)):
                hamming_dist = calc_hamming_distance(all_hv_in_class[k], hv_per_class[j])
                hamming_dist_per_hv.append(hamming_dist)
                min_dist_across_class = min(min_dist_across_class, hamming_dist)
            hamming_dist_list.append(int(np.mean(hamming_dist_per_hv)))
        hamming_dist_across_class[i] = hamming_dist_list

    logging.debug(f"Min Hamming distance across classes: {min_dist_across_class}")
    suggested_threshold = min_dist_across_class
    logging.info(f"Average Hamming distance per class: {average_hamming_dist_per_class}")
    logging.info(f"Hamming distance from a class to other classes: {hamming_dist_across_class}")
    # expect the distance within each class is small but big across classes
    # it needs to be analyzed by checking average_hamming_dist_per_class and hamming_dist_across_class
    # it currently is suggested to at 450
    return hv_per_class_dict, hv_all_classes, suggested_threshold

def run_hdc_inference(
        hv_per_class_dict: dict,
        testcase_folder: str = "./",
        testcase: str = "C_Easy1_noise005.mat",
        fs: int = 24000,
        inference_start_time: float = 1.0,
        inference_end_time: float = 2.0,
        inference_window_length: int = 30,
        with_front_end: bool = False,
        front_end_mode: str = 'NEO_DVT',
        filter_params: dict = {'filter_en' : True,'corner_freq':[1000,6000],'order' : 1,'axis' : 0}
        ):
    """! Run HDC inference on the given hypervectors.
    @param hv_per_class_dict: Hypervectors for each class (include "zero" class)
    @param testcase_folder: Folder containing the test case files
    @param testcase: Name of the test case file
    @param fs: Sampling frequency
    @param inference_start_time: Inference start time
    @param inference_end_time: Inference end time
    @param inference_window_length: Inference window length
    @param with_front_end: If with front end precisely monitoring spikes
    @param front_end_mode: Choose the front_end mode to detect the spikes
    """
    # load hv from files
    IM = load_hypervector(f"{testcase_folder}IM_hv.txt")  # for signal level
    EM = load_hypervector(f"{testcase_folder}EM_hv.txt")  # for signal timing

    # load the inference sample
    matlab_source_data = sio.loadmat(f"{testcase_folder}{testcase}")
    recording_traces = matlab_source_data["data"][0]
    recording_traces = filtering(recording_traces, filter_params,fs) 
    spike_class = matlab_source_data["spike_class"][0][0][0]

    if front_end_mode == 'ground_truth':
        spike_times = matlab_source_data["spike_times"][0][0][0]
        spike_times += 24  # 24 is for label correction
    elif front_end_mode =='NEO_DVT':
        testcase_name = os.path.splitext(testcase)[0]
        NEO_DVT_file = f'neo_dvt/neodvt_result/spike_instants{testcase_name}.csv'
        spike_times = np.loadtxt(NEO_DVT_file, delimiter=",", dtype=int)    
    

    # inference
    logging.info("Inference...")

    inference_sequence = recording_traces[int(fs * inference_start_time): int(fs * inference_end_time)]
    num_of_inference_windows = len(inference_sequence) // inference_window_length
    inference_hv: list = []  # record of hypervectors of different windows
    inference_spike_labels: list = []  # record of spike labels for different windows (reference)
    inference_spike_class: list = []  # record of spike class for different windows (reference)
    inference_spike_labels_inferred: list = []  # record of spike labels for different windows (inferred)
    inference_spike_class_inferred: list = []  # record of spike class for different windows (inferred)

    # generate spike labels for windows (reference)
    spike_times_to_end = spike_times[spike_times <= int(fs * inference_end_time)]
    spike_times = spike_times_to_end[spike_times_to_end >= int(fs * inference_start_time)]
    window_having_spike = [int((t - fs * inference_start_time) // inference_window_length) for t in spike_times]
    inference_spike_labels = [1 if i in window_having_spike else 0 for i in range(num_of_inference_windows)]

    # check the minimal diff of spike time (in reference)
    spike_times_diff = [spike_times[i+1] - spike_times[i] for i in range(len(spike_times) - 1)]
    spike_times_diff_min = np.min(spike_times_diff) if spike_times_diff else 0
    if spike_times_diff_min < inference_window_length:
        logging.warning(f"Spike times ({spike_times_diff_min}) are too close to the window length "
                        f"{inference_window_length}, may cause issues in inference.")

    # generate spike class for windows (reference)
    spike_class = spike_class[0: len(spike_times_to_end)]
    spike_class = spike_class[-len(spike_times):]
    assert len(spike_class) == len(spike_times), "Spike class length does not match spike times length."
    inference_spike_class = [int(spike_class[window_having_spike.index(i)] if i in window_having_spike else 0)
                             for i in range(num_of_inference_windows)]

    ######################################################
    ## To be replaced with real frontend processing alg.
    # add frontend processing (only keep spike windows)
    if with_front_end:
        inference_sequence_processed = np.array([])
        for tim in spike_times:
            seq_clipped = recording_traces[tim - inference_window_length//2: tim + inference_window_length//2]
            inference_sequence_processed = np.append(inference_sequence_processed, seq_clipped)
        inference_spike_labels = np.ones(len(inference_sequence_processed) // inference_window_length)
        inference_spike_class = spike_class
        assert len(inference_spike_class) == len(inference_spike_labels), \
            f"Inference spike class and labels must have the same length."
    else:
        inference_sequence_processed = inference_sequence
        inference_spike_labels = inference_spike_labels
        inference_spike_class = inference_spike_class
    ######################################################

    # generate hypervectors for inference windows
    logging.info("S1: Calculate hypervectors per window for inference...")
    inference_hv = calc_hdc_of_windows(
        training_sequence=inference_sequence_processed,
        IM=IM,
        EM=EM,
        training_window_length=inference_window_length,
    )

    # similarity check for each window
    logging.info("S2: Check similarity for each window...")
    for i in tqdm.tqdm(range(len(inference_hv)), ascii="░▒▓█", desc="Inference windows"):
        # calculate similarity with each class
        similarities = {}
        for class_name, class_hv in hv_per_class_dict.items():
            hamming_distance = calc_hamming_distance(inference_hv[i], class_hv)
            similarities[class_name] = hamming_distance

        # find the class with the highest similarity
        inferred_class = int(min(similarities, key=similarities.get))
        inference_spike_labels_inferred.append(1 if inferred_class != 0 else 0)
        inference_spike_class_inferred.append(inferred_class)

    # compare with the reference and count the correct/incorrect classifications
    num_correct_spike_inferred = sum(1 for i in range(len(inference_spike_labels_inferred))
                      if inference_spike_labels_inferred[i] == inference_spike_labels[i])
    num_incorrect_spike_inferred = len(inference_spike_labels_inferred) - num_correct_spike_inferred
    correct_identify_ratio = num_correct_spike_inferred / len(inference_spike_labels_inferred)

    # num_correct_class_inferred: include both zero class and non-zero classes
    num_correct_class_inferred = sum(1 for i in range(len(inference_spike_class_inferred))
                                      if inference_spike_class_inferred[i] == inference_spike_class[i])
    num_incorrect_class_inferred = len(inference_spike_class_inferred) - num_correct_class_inferred
    class_identify_ratio = num_correct_class_inferred / len(inference_spike_class_inferred)

    logging.info(f"Correct spike identifications: {num_correct_spike_inferred} ({correct_identify_ratio:.2%})")
    logging.info(f"Correct class identifications: {num_correct_class_inferred} ({class_identify_ratio:.2%})")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s")

    # parameters
    regenerate_hypervector = False
    with_front_end = True
    front_end_mode = 'NEO_DVT' #choose from NEO_DVT or ground_truth
    filter_params = {
        'filter_en' : True,
        'corner_freq': [1000,6000],
        'order' : 1,
        'axis' : 0
    }

    hv_length = 1024
    fs = 24000 # sampling frequency (Hz)
    training_time = 0.5 # seconds
    training_window_length = 30 # number of samples
    testcase_folder = "./quiroga/"
    testcase_list = [
        "C_Easy1_noise005.mat",
        #"C_Easy2_noise005.mat",
        #"C_Difficult2_noise015.mat",
                    ]

    for testcase in testcase_list:
        hv_per_class_dict, hv_all_classes, suggested_threshold = run_hdc(
                regenerate_hypervector=regenerate_hypervector,
                testcase_folder=testcase_folder,
                testcase=testcase,
                training_window_length=training_window_length,
                fs=fs,
                training_time=training_time,
                hv_length=hv_length,
                with_front_end=with_front_end,
                filter_params=filter_params
                )
        run_hdc_inference(
            hv_per_class_dict=hv_per_class_dict,
            testcase_folder=testcase_folder,
            testcase=testcase,
            fs=fs,
            inference_start_time=training_time,
            inference_end_time=training_time + 10,
            inference_window_length=training_window_length,
            with_front_end=with_front_end,
            front_end_mode=front_end_mode, 
            filter_params=filter_params
        )
        breakpoint()
