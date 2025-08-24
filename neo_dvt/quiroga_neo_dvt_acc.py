#from sklearn.cluster import KMeans
import sys
sys.path.append("./src")
import numpy as np
import spikeinterface.extractors as se
import argparse
# local functions
from yz_src import readin,detection, comparing_result,filtering
import os
import pandas as pd
# get current directory
workingdir = os.getcwd()
print("current path:", workingdir)

# make sure response folder exist
neodvt_dir = os.path.join(workingdir, "neodvt_result")
os.makedirs(neodvt_dir, exist_ok=True)


parser = argparse.ArgumentParser(description="Run accuracy calculation.")
parser.add_argument("--emphasizer_type", type=str, required=True, help="Type of emphasizer (e.g., 'delta')")
args = parser.parse_args()



detecting_type = "D" # "D"-detection or "E"-extraction
comparing_method = 0 # 0=>Spikeinterface; 1=> accuracy score (only valid for non-detection)

thr_factor_list = [2.45]
print(thr_factor_list)

plot_on = 0
filter_en = 0
delta_time = 0.4

print_seperate_acc = False
quiroga_dir = "/imec/other/macaw/projectdata/quiroga_datasets/"
# Set the emphasizer_type dynamically
emphasizer_type = args.emphasizer_type


detection_type = 'DVT' #DVT or traditional

if detection_type == 'DVT':
    spike_duration = 0.5 #ms
else:
    spike_duration = 1.5 #ms

print(emphasizer_type,'-', detection_type, '-', delta_time,'-filter',filter_en,'-spike_dur',spike_duration)

quiroga_data_list = [
                     f"{quiroga_dir}C_Easy1_noise005.mat", \
                     f"{quiroga_dir}C_Easy1_noise01.mat", \
                     f"{quiroga_dir}C_Easy1_noise015.mat", \
                     f"{quiroga_dir}C_Easy1_noise02.mat", \
                     f"{quiroga_dir}C_Easy2_noise005.mat", \
                     f"{quiroga_dir}C_Easy2_noise01.mat", \
                     f"{quiroga_dir}C_Easy2_noise015.mat", \
                     f"{quiroga_dir}C_Easy2_noise02.mat", \
                     f"{quiroga_dir}C_Difficult1_noise005.mat", \
                     f"{quiroga_dir}C_Difficult1_noise01.mat", \
                     f"{quiroga_dir}C_Difficult1_noise015.mat", \
                     f"{quiroga_dir}C_Difficult1_noise02.mat", \
                     f"{quiroga_dir}C_Difficult2_noise005.mat", \
                     f"{quiroga_dir}C_Difficult2_noise01.mat", \
                     f"{quiroga_dir}C_Difficult2_noise015.mat", \
                     f"{quiroga_dir}C_Difficult2_noise02.mat"]

# detection parameter dicts
filter_params = {
    'filter_en' : filter_en,
    'corner_freq': [1000,6000],
    'order' : 1,
    'axis' : 0
}
detection_params = {
    'detection_type': detection_type,  # DVT/traditional
    'emphasizer_type': emphasizer_type, # raw/NEO/ED/delta
    'window': int(24*spike_duration),  #window for masking detection point in time #quiroga-0.5ms-?12 points
    'detect_canceling_window': [0,int(24*spike_duration) ],
    'online_mode':False,

    'thr_factor': 2.5,
    'LC_mode':False,
    'min_thr':None,
    'points_for_std':32768   #(2^15)
    }
data_list = ['Quiroga','MEArec','NP']
data_type = data_list[0]
length_of_recording = 60 #s    

set_name_list = ["set"]* len(quiroga_data_list)


for x, thr_factor in enumerate(thr_factor_list):
    detection_params['thr_factor']= thr_factor
    accuracy_with_given_thr_factor = [0]*len(quiroga_data_list)
    detected = [0]*len(quiroga_data_list)
    fp = [0]*len(quiroga_data_list)
    fn = [0]*len(quiroga_data_list)
    for index, recording_path in enumerate(quiroga_data_list):
        
        set_name = quiroga_data_list[index].split(quiroga_dir)[-1][:-4]
        set_name_list[index] = set_name 
        gt_path = recording_path
        [fs, coords, recording_traces, GT_instants, GT_labels,GT_coords,centering_positions,_] = readin.readin(recording_path,gt_path, length_of_recording,data_type,set_name=index)
        GT = se.NumpySorting.from_times_labels(np.array(GT_instants), np.array(GT_labels), fs)

        #filtering
        recording_traces = filtering.filtering(recording_traces, filter_params,fs) 

        if(detecting_type == 'D'):
            [spike_instants,spike_coords,recording_emphasized,_]=detection.spike_detection(recording_traces,detection_params) #inside, print detection results
            cmp_detection = comparing_result.detect_acc_calculation(GT_instants, spike_instants,fs,delta_time)
            tmp_accuracy,tmp_accuracy_pd = comparing_result.accuracy_print (set_name, cmp_detection, comparing_method, print_pd = print_seperate_acc)
            accuracy_with_given_thr_factor[index] = tmp_accuracy
            detected[index] = tmp_accuracy_pd['num_tested'][0]
            fp[index] = tmp_accuracy_pd['fp'][0]
            fn[index] = tmp_accuracy_pd['fn'][0]
            print(f'for dataset{set_name},acc =  {tmp_accuracy* 100: .2f}%, detected = {detected[index]}, fp = {fp[index]} , fn(miss) = {fn[index]} ')
            spike_instants_file_path = os.path.join(neodvt_dir, f"spike_instants{set_name}.csv")
            np.savetxt(spike_instants_file_path, spike_instants, fmt="%d", delimiter=",")
        else:
            spike_instants = np.array(GT_instants)
            spike_coords = np.array(GT_coords)
    
    
    if(detecting_type == 'D'):
        avg_acc = np.average(accuracy_with_given_thr_factor)
        avg_detected = np.average(detected)
        avg_fp = np.average(fp)
        avg_fn = np.average(fn)
        print(f"with thr_factor {thr_factor},: acc =  {avg_acc* 100: .2f}%, detected = {avg_detected}, fp = {avg_fp} , fn(miss) = {avg_fn} ")


accuracy_with_given_thr_factor = np.array(accuracy_with_given_thr_factor, dtype=float)
fp = np.array(fp, dtype=int)
fn = np.array(fn, dtype=int)
detected = np.array(detected, dtype=int)

# get DataFrame
df = pd.DataFrame({
    "set_name": set_name_list,
    "acc": accuracy_with_given_thr_factor,
    "fp": fp,
    "fn": fn,
    "detected": detected
})

print(df)
quiroga_acc_path = os.path.join(neodvt_dir, f"neodvt_acc.csv")
df.to_csv(quiroga_acc_path, index=False)

