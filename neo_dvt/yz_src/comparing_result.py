import spikeinterface.extractors as se
import spikeinterface.comparison as sc
import numpy as np
from sklearn.metrics import accuracy_score
import pdb
def detect_acc_calculation(GT_instants, spike_instants,fs,delta_time = None):
    # function: calculate detection accuracy and print it, defalt with pd
    GT_fake_labels = np.zeros_like(GT_instants)
    sorting_fake_labels =  np.zeros_like(spike_instants)
    GT_detection = se.NumpySorting.from_times_labels(np.array(GT_instants), np.array(GT_fake_labels), fs)
    sorting_detection = se.NumpySorting.from_times_labels(np.array(spike_instants), np.array(sorting_fake_labels), fs)
    cmp_detection = sc.compare_sorter_to_ground_truth(GT_detection, sorting_detection, gt_name = "gt", tested_name = "sorted", delta_time = delta_time, match_score = 0.0)
    #accuracy_print('detection', cmp_detection, comparing_method = 0, print_pd = True)
    return cmp_detection

def tot_acc_calculation(GT_instants, GT_labels, spike_instants,sorting_labels,fs,train_set= None,comparing_method = 0, print_pd = False):
    #train_set: integer or percentage
    # Function: using GT/sorting labels & instants to give the accuracy results. 
    # if train_set = None, only cmp_total is calculated
    # if not, results are divided into trainset and test set for comparison
    GT_object = se.NumpySorting.from_times_labels(np.array(GT_instants), np.array(GT_labels), fs)
    sorting_object = se.NumpySorting.from_times_labels(np.array(spike_instants), np.array(sorting_labels), fs)
    if comparing_method == 0:
        if train_set == None: #no_splitting
            cmp_all = sc.compare_sorter_to_ground_truth(GT_object, sorting_object, delta_time = 0.4, gt_name = "gt", tested_name = "sorted", match_mode = 'best',match_score = 0.0,exhaustive_gt= True)
            cmp_train = None
            cmp_test = None
            accuracy_print('all', cmp_all, comparing_method,print_pd)
        else:
            if train_set <1: #percentage
                train_set = np.int(train_set*len(GT_instants))
            # train_set >1: #exact train_set number
            sorting_labels_train = sorting_labels[:train_set]
            sorting_labels_test = sorting_labels[train_set:]
            spike_instants_train = spike_instants[:train_set]
            spike_instants_test = spike_instants[train_set:]
            sort_train = se.NumpySorting.from_times_labels(np.array(spike_instants_train), np.array(sorting_labels_train), fs)
            sort_test = se.NumpySorting.from_times_labels(np.array(spike_instants_test), np.array(sorting_labels_test), fs)
            GT_labels_train = GT_labels[:train_set]
            GT_labels_test = GT_labels[train_set:]
            GT_instants_train = GT_instants[:train_set]
            GT_instants_test = GT_instants[train_set:]
            GT_train = se.NumpySorting.from_times_labels(np.array(GT_instants_train), np.array(GT_labels_train), fs)
            GT_test = se.NumpySorting.from_times_labels(np.array(GT_instants_test), np.array(GT_labels_test), fs)

            cmp_all = sc.compare_sorter_to_ground_truth(GT_object, sorting_object, gt_name = "gt", tested_name = "sorted", match_score = 0.0)
            cmp_train = sc.compare_sorter_to_ground_truth(GT_train, sort_train, gt_name = "gt", tested_name = "sorted", match_score = 0.0)
            cmp_test = sc.compare_sorter_to_ground_truth(GT_test, sort_test, gt_name = "gt", tested_name = "sorted", match_score = 0.0)
            #accuracy_print('all', cmp_all, comparing_method,print_pd)
            #accuracy_print('train', cmp_train, comparing_method,print_pd)
            #accuracy_print('test', cmp_test, comparing_method,print_pd)
    elif comparing_method ==1:
        cmp_train = None
        cmp_test = None
        cmp_all = sc.compare_sorter_to_ground_truth(GT_object, sorting_object, gt_name = "gt", tested_name = "sorted", match_score = 0.0)
        df = cmp_all.get_performance(method = "raw_count") 
        tested_ids = df['tested_id'].tolist()
        GT_labels_reordered = np.zeros(len(GT_labels),int) 
        for i in range(len(GT_labels)):
            for j in range(len(np.unique(GT_labels))):
                if GT_labels[i] == j+1:
                    GT_labels_reordered[i] = tested_ids[j]

        
        #accuracy all
        tmp_accuracy3 = accuracy_score(GT_labels_reordered, sorting_labels)
        #accuracy all - from SI only 
        tp = df['tp'].tolist()
        gt = df['num_gt'].tolist()
        tmp_accuracy4 = sum(tp)/sum(gt)
        print(f" Accuracy all_acc_score:{tmp_accuracy3 * 100: .2f} %", end = '    '  )
        print(f" all_si_acc_score: {tmp_accuracy4 * 100: .2f} %", end = '    ' )


        if train_set != None: 
            if train_set <1: #percentage
                train_set = np.int(train_set*len(GT_instants))
            #accuracy train
            tmp_accuracy1 = accuracy_score(GT_labels_reordered[:train_set], sorting_labels[:train_set])
            print(f" train: {tmp_accuracy1 * 100: .2f} %", end = '    ')
            #accuracy test
            tmp_accuracy2 = accuracy_score(GT_labels_reordered[train_set:], sorting_labels[train_set:])
            print(f" test: {tmp_accuracy2 * 100: .2f} %", end = '    ' )
    return cmp_all,cmp_train,cmp_test

def accuracy_print (name, cmp_object, comparing_method,print_pd):
    cmp_method = ("raw_count", "by_unit", "pooled_with_average")
    tmp_accuracy = cmp_object.get_performance(method = cmp_method[2])["accuracy"]
    tmp_accuracy_pd = cmp_object.get_performance(method = cmp_method[0])

    if print_pd == True:
        fp = tmp_accuracy_pd['fp'][0]
        fn = tmp_accuracy_pd['fn'][0]
        tp = tmp_accuracy_pd['tp'][0]
        print(f"{name} \t accuracy ={tmp_accuracy * 100: .2f}%, tp = {tp}, fp = {fp}, fn = {fn} ")
        #print(tmp_accuracy_pd)
    #else:
        #print(f"{name} accuracy ={tmp_accuracy * 100: .2f} %   ") # next line
        #print(f"{name} accuracy ={tmp_accuracy * 100: .2f} %   ",end = "")
        
    return tmp_accuracy,tmp_accuracy_pd


def my_detect_acc_multi(channel_count,GT_instants,GT_coords, GT_labels, spike_instants,spike_coords,fs,delta_time = 0.4):
    ##################parameters########################
    d_wind_lines = 6
    delta_length= int(fs*delta_time/1000)
    wind_p_check = [-delta_length,delta_length]
    missed_spike_instants = []
    redundant_spikes = []

    #missed_spike_instants = np.zeros([len(GT_instants), 2])
    if channel_count == 1:
        spike_coords = np.zeros_like(spike_instants)
    # function: calculate detection accuracy and print it
    GT_spike_information = np.stack((GT_instants,GT_coords), axis=1)
    a_spike_instants_check = np.stack((spike_instants,spike_coords), axis=1)
    a_spike_instants = np.zeros([len(spike_instants), 4])
    a_spike_instants[:,0:2] = a_spike_instants_check #a_spike_instants: 3列信息，其中第一列为spike instants,第二列为spike coords， 第三列为GT_coords, 第四列为（如有）对应的GT instants, (如无)0

    for n in range(len(GT_instants)):
        c = GT_coords[n]
        t = GT_instants[n]
        l_c = max(0, c - 2 * d_wind_lines - c % 2)
        r_c = min(channel_count-1, c + 2 * d_wind_lines + (c+1) % 2)
        tmp_index = np.where(
            (a_spike_instants_check[:, 0] >= GT_spike_information[n, 0] + wind_p_check[0]) &
            (a_spike_instants_check[:, 0] <= GT_spike_information[n, 0] + wind_p_check[1]) &
            (a_spike_instants_check[:, 1] >= l_c) &
            (a_spike_instants_check[:, 1] <= r_c)
        )[0]
        if len(tmp_index) ==0:
            # no matched spikes, the spike is missed
            #missed_spike_instants[n,:] = GT_spike_information[n, :] # put the missed spike in the matrix
            missed_spike_instants.append(GT_spike_information[n, :])
        else:
            tmp_min = tmp_index[0]
            if len(tmp_index) >1:

                for time in range(1,len(tmp_index)):
                    if (abs(a_spike_instants_check[tmp_index[time], 0] - t) <abs(a_spike_instants_check[tmp_min, 0] - t)):
                        tmp_min = tmp_index[time]
                for x in tmp_index:
                    if x != tmp_min:
                        redundant_spikes.append(a_spike_instants_check[x]) #没做完，因为 要去掉那个tmin
            a_spike_instants_check[tmp_min,0] = 0 #delete the spike from the possible solutions
            a_spike_instants[tmp_min,2] = c # centering coords
            a_spike_instants[tmp_min,3] = GT_labels[n]+1 #+1是为了防止原始GT labels从0计算
            
    n_correct_spikes   = np.sum(a_spike_instants[:, 3] > 0)
    fp_spikes_ind = np.where(a_spike_instants[:, 3] == 0)[0]
    false_positive_spikes = a_spike_instants[fp_spikes_ind][:,0:3]

    n_incoming_spikes = len(GT_instants)
    n_d_spike_instants = len(spike_instants)
    n_missed_spikes    = n_incoming_spikes - n_correct_spikes
    n_false_pos_spikes = n_d_spike_instants - n_correct_spikes  # False positive spikes can be noise or duplicated spikes
    det_accuracy       = n_correct_spikes / (n_incoming_spikes + n_false_pos_spikes)     
    #print('missed',missed_spike_instants)
    #print('redundant',redundant_spikes)
    #print('fp',false_positive_spikes)
    return det_accuracy,n_correct_spikes,n_incoming_spikes,n_missed_spikes,n_false_pos_spikes,a_spike_instants.astype(int)

def array_to_dict(array):
    result = {}
    for row in array:
        key = row[3]
        value = row[:3]
        if key in result:
            result[key].append(value)
        else:
            result[key] = [value]
    return result

def print_labels_and_coords(result_dict):
    for key, values in sorted(result_dict.items()):
        second_column_values = sorted(list(set(value[1] for value in values)))
        print(f"label-{key} \t center-{values[0][2]} \t detected on:  {second_column_values}")

from collections import Counter
def print_detection_splitness(result_dict,print_en = 0):
    max_percentage_element_list = []
    for key, values in sorted(result_dict.items()):
        if print_en ==1:
            print('label-', key, '\t center-', values[0][2], end='\t')
        element_counts = dict(Counter(np.array(values)[:,1]))

        total_elements = len(np.array(values)[:,1])

        # 计算每个元素的百分比
        element_percentage = {element: (count / total_elements) for element, count in element_counts.items()}
        #print(element_percentage)
        top_three_percentages = dict(sorted(element_percentage.items(), key=lambda item: item[1], reverse=True)[:1])
        #print(top_three_percentages)

        element_count_arr = np.array(list(element_counts.items())) #只有个数，具体是哪个通道被忽略了
        max_index = np.argmax(element_count_arr[:,1])
        max_coord = element_count_arr[max_index][0]
        max_percentage = element_count_arr[max_index][1]/total_elements
        if print_en ==1:
            print('on \t', max_coord, 'with \t',"{:.2%}".format(max_percentage) )
        max_percentage_element_list.append([max_coord,max_percentage])
        #print(values)
    return max_percentage_element_list