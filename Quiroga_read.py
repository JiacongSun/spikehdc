#from sklearn.cluster import KMeans
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

#[fs, coords, gt_instants,gt_labels] = readin(recording_path,gt_path, length_of_recording,data_type)
# coords: x/y information
# recording_traces: length*coords, numpy, raw recording; index follows the same as Bernardo's (x-x=y-y...)
# sortingGT: sorting objects, with instants(time) and labels (ID)
def readin (recording_path,gt_path, length_of_recording,data_type,set_name = None):
    if data_type == "Quiroga":
        quiroga_dir = r"./" # change this to your folder
        quiroga_data_list = [
                     f"{quiroga_dir}C_Easy1_noise005.mat", \
                     #f"{quiroga_dir}C_Easy1_noise01.mat", \
                     #f"{quiroga_dir}C_Easy1_noise015.mat", \
                     #f"{quiroga_dir}C_Easy1_noise02.mat", \
                     f"{quiroga_dir}C_Easy2_noise005.mat", \
                     #f"{quiroga_dir}C_Easy2_noise01.mat", \
                     #f"{quiroga_dir}C_Easy2_noise015.mat", \
                     #f"{quiroga_dir}C_Easy2_noise02.mat", \
                     #f"{quiroga_dir}C_Difficult1_noise005.mat", \
                     #f"{quiroga_dir}C_Difficult1_noise01.mat", \
                     #f"{quiroga_dir}C_Difficult1_noise015.mat", \
                     #f"{quiroga_dir}C_Difficult1_noise02.mat", \
                     #f"{quiroga_dir}C_Difficult2_noise005.mat", \
                     #f"{quiroga_dir}C_Difficult2_noise01.mat", \
                     #f"{quiroga_dir}C_Difficult2_noise015.mat", \
                     #f"{quiroga_dir}C_Difficult2_noise02.mat"
                     ]
        if (set_name>=0 & set_name<16):
            recording_path = quiroga_data_list[set_name]
        #print('read quiroga')
        new_coords = [0,0]
        fs =24000
        mat_data = sio.loadmat(recording_path)
        recording_traces = np.array(mat_data["data"][0])
        recording_traces = recording_traces[0:int(length_of_recording*fs)]

        gt_instants = mat_data["spike_times"][0][0][0]
        gt_instants = [gt_instants for gt_instants in gt_instants if gt_instants < int(length_of_recording*fs)]

        gt_labels = mat_data["spike_class"][0][0][0]
        gt_labels = gt_labels[0:len(gt_instants)]
        gt_c_coords = np.zeros([len(gt_instants)],int)
        centering_positions = None
        # quiroga operation: align gt_labels with ground truth (dont find max, will cause easily overlapped data wrong)
        #gt_instants = np.array(gt_instants) +30 #+26 for LC bit test
        gt_instants = np.array(gt_instants)+24


    """
    @fs: Sampling frequency
    @new_coords: [0, 0], not used
    @recording_traces: Recorded signals
    @gt_instants: A timing record of the ground truth spike, which provides the truth reference
    @gt_labels: The labels corresponding to the ground truth spikes
    @gt_c_coords: not used
    @centering_positions: not used
    """
    return fs, new_coords, recording_traces, gt_instants, gt_labels, gt_c_coords, centering_positions

def plot(recording_traces, GT_instants, GT_labels, fs, time_to_plot, set_name=None):
    samples_to_plot = fs * time_to_plot
    tt = np.arange(0, samples_to_plot) / fs
    plt.figure(figsize=(20, 6))

    plt.plot(tt, recording_traces[0:int(samples_to_plot)])


    gt_times = GT_instants / fs
    mask = (gt_times >= 0) & (gt_times <= time_to_plot)
    gt_times = gt_times[mask]
    gt_labels = GT_labels[mask]


    unique_labels = np.unique(gt_labels)
    cmap = plt.cm.get_cmap('Set1', len(unique_labels))

    
    for i, lbl in enumerate(unique_labels):
        idx = gt_labels == lbl
        plt.scatter(gt_times[idx], np.ones(np.sum(idx)), 
                    c=[cmap(i)], s=20, zorder=3, label=f"Label {lbl}")

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'raw_trace_{set_name}.png')
    plt.close()

if __name__ == "__main__":
    """! function from yz"""
    recording_path = None
    gt_path = recording_path
    length_of_recording = 60
    data_type = 'Quiroga'

    time_to_plot = 0.5  # up to length_of_recording
    for i in range(2):
        result = readin(recording_path=recording_path,
                                gt_path=gt_path,
                                length_of_recording=length_of_recording,
                                data_type=data_type,
                                set_name=i)
        fs, coords, recording_traces, GT_instants, GT_labels, GT_coords, centering_positions = result

        plot(recording_traces=recording_traces,
                    GT_instants=GT_instants,
                    GT_labels=GT_labels,
                    fs=fs,
                    time_to_plot=time_to_plot,
                    set_name=i)
