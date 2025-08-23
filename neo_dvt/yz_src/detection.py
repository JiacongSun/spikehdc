#[spike_instants,spike_coords]=spike_detection(recording_traces)
import matplotlib.pyplot as plt
import numpy as np
import pdb
def alignment(recording_traces,spike_not_aligned_instants,params):
    align_mode = params['align_mode']
    align_canceling_window = params['align_canceling_window']
    spike_length = [0,align_canceling_window[1]]
    instants_aligned = []
    for instant in spike_not_aligned_instants:
        start = instant+spike_length[0]
        end = instant+spike_length[1]
        window_data = recording_traces[start:end] # windowing
        if (align_mode == 0): #max abs
            index = np.argmax(np.abs(window_data))+start 
        elif (align_mode ==1): #max slope
            index = np.argmax(np.diff(window_data))+start
        instants_aligned.append(index) 
        #pdb.set_trace()
    spike_instants = instants_aligned
    return spike_instants

def emphasizer(arr,params):
    arr = np.array(arr)
    emphasizer_type = params["emphasizer_type"]
    if emphasizer_type == "NEO":
        padded_x = np.concatenate(([arr[0]], arr, [arr[-1]]))
        result = padded_x[1:-1] ** 2 - padded_x[:-2] * padded_x[2:]
    elif emphasizer_type == "ED":
        padded_x = np.concatenate(([arr[0]], arr, [arr[-1]]))
        result = np.diff(padded_x) #x(n) - x(n-1)
        result = np.square(result) # square
        result = result[:-1] #size the same as arr, so the first number is always 0. 
    elif emphasizer_type == "delta":
        padded_x = np.concatenate(([arr[0]], arr, [arr[-1]]))
        result = np.diff(padded_x)
        result = result[:-1] #size the same as arr, so the first number is always 0.
    elif emphasizer_type == "raw":
        result = arr
    return result

def LC_spatio_detection(recording_traces,thr,params):
    detection_type = params["detection_type"]
    detect_canceling_window = params["detect_canceling_window"]
    # choose type for detection, and get the matrix
    d_spike_matrix = np.zeros_like(recording_traces)
    for c in range(recording_traces.shape[1]): 
        arr = recording_traces[:,c]
        if detection_type == "DVT":
            d_spike_matrix[:,c]  = DVT_detection(arr,thr[c])
        elif detection_type == "traditional":    
            d_spike_matrix[:,c]  = traditional_detection(arr,thr[:,c])
    channel_count = recording_traces.shape[1]
    
    # canceling in the time/spatio window (generate d_spike_matrix)
    d_wind_lines = detect_canceling_window[0] #2*(6*2+1) = 26个neighbor
    d_wind_points = detect_canceling_window[1] # 之前是20
    d_spike_rows = np.where(np.any(d_spike_matrix != 0, axis=1))[0] # find the time where at least 1 coords have spikes
    for n in d_spike_rows:
        for c in range(recording_traces.shape[1]):
            if d_spike_matrix[n, c] == 1:
                l_c = max(0, c - 2 * d_wind_lines - c % 2)
                r_c = min(channel_count-1, c + 2 * d_wind_lines + (c+1) % 2)
                d_spike_matrix[n : n + d_wind_points + 1, l_c:r_c + 1] = 0
                d_spike_matrix[n, c] = 1

    spike_instants, spike_coords = np.where(d_spike_matrix == 1)
    return spike_instants, spike_coords
    # return time instant and coords


def detection_1D(arr,thr,params,stride = None):
    if stride == None:
        stride = 1
    detection_type = params["detection_type"]
    
    detect_canceling_window = int(params["detect_canceling_window"][1])
    # choose type for detection
    if detection_type == "DVT":
        detected_spikes = DVT_detection(arr,thr)
    elif detection_type == "traditional":
        detected_spikes = traditional_detection(arr,thr)
    # canceling in the window
    for i in range(len(detected_spikes)):
        if detected_spikes[i] == 1:
            detected_spikes[i+1:i+detect_canceling_window+1] = [0]*detect_canceling_window
    # spike alignment
    
    # translate to spike_instants
    spike_instants = np.where(np.array(detected_spikes) == 1)[0]
    return spike_instants

def DVT_detection(vector,thr):
    n = len(vector)
    detected_spikes = [0] * n  
    abs_vector = abs(vector)
    arr = vector
    for i in range(1, n-1):
        #if abs_vector[i] > abs_vector[i-1] and abs_vector[i] > abs_vector[i+1] and abs_vector[i]>thr[i] :
        #    detected_spikes[i] = 1
        #considering plateu
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i]>thr[i]:
            detected_spikes[i] = 1
        elif arr[i] == arr[i + 1] and arr[i] > arr[i - 1] and arr[i]>thr[i]:
            # Handle cases where the current element is part of a plateau
            j = i + 1
            while j < len(arr) and arr[j] == arr[i]:
                j += 1
            if j < len(arr) and arr[j] < arr[i]:
                detected_spikes[i] = 1
    return detected_spikes

def traditional_detection(vector,thr):
    n = len(vector)
    detected_spikes = [0] * n  
    abs_vector = abs(vector)
    for i in range(0, n):
        if abs_vector[i]>thr[i] :
            detected_spikes[i] = 1
    return detected_spikes

def thr_comp(arr,params,LC_window = None,thr_predefined = None,):
    online_mode = params["online_mode"]
    thr_factor = params["thr_factor"]
    LC_mode = params['LC_mode']
    min_thr = params['min_thr']
    


    thr = np.zeros_like(arr)

    if (LC_mode == False): #normal detect
        points_for_std = params["points_for_std"] #enabled when LC_mode is false 
        if (online_mode == True):
            num_points = len(arr) 
            for i in range(0, num_points, points_for_std):
                sub_array = arr[i:i + points_for_std]
                #std_dev = np.std(sub_array)
                std_dev = np.mean(np.abs(sub_array)) #before *1.5
                thr[i:i + points_for_std] = std_dev*thr_factor

        else:
            std_traces = np.std(arr)
            thr_cte = thr_factor*std_traces
            thr[:] = thr_cte
    else:
        [LC_exponents_start, LC_exponents_end] = params['LC_exponent']
        points_for_LC_avg = 2**LC_exponents_end
        #points_for_LC_avg = params['points_for_LC_avg'] #enabled when LC_mode is true 
        #points_for_LC_avg_start = 128 #enabled when LC_mode is true 
        #if (online_mode == True):
        #    num_points = len(arr) 
        #    for i in range(0, num_points, points_for_LC_avg):
        #        sub_array = arr[i:i + points_for_LC_avg]
        #        #std_dev = np.std(sub_array)
        #        std_dev = np.mean(np.abs(sub_array)) #before *1.5
        #        thr[i:i + points_for_LC_avg] = std_dev*thr_factor
        #
        if (online_mode == True):
            #######################initialize thr,updated per 2^x till the constant, use the saved pulses for generation ############################
            # to avoid the problem of start errors, we define the start mean pulse number in a given window to be larger than 0.5
            num_points = len(arr) 
            thr[:2**LC_exponents_start] = 0
            for i in range (LC_exponents_start,LC_exponents_end):
                sub_arr = arr[:2**i]
                if(params['thr_mode'] == 0):
                    std_dev = np.mean(sub_arr)*LC_window

                else:
                    std_dev = np.sqrt(np.mean(sub_arr)*LC_window)                
                #print(std_dev)

                thr[2**i:2**(i+1)] = max(min_thr,std_dev*thr_factor)
            ########################other thr, use the sub array of 2^M as indicator for threshold#############################
            for i in range(0, num_points, points_for_LC_avg):
                sub_array = arr[i:i + points_for_LC_avg]
                #std_dev = np.std(sub_array)
                if(params['thr_mode'] == 0):
                    std_dev = np.mean(sub_array)*LC_window
                else:
                    std_dev = np.sqrt(np.mean(sub_array)*LC_window)
                #print('std_dev',std_dev)
                #pdb.set_trace()
                thr[i+points_for_LC_avg:i + 2*points_for_LC_avg] = max(min_thr,std_dev*thr_factor)
            #plt.plot(thr)
            #plt.show()
            #pdb.set_trace()
                #print('thr',thr)
        else:
            thr[:] = thr_factor*np.sqrt(np.mean(arr)*LC_window)
            thr[:] = thr_predefined
            #thr[:] = thr_factor*np.mean(arr)*LC_window
            #thr[:] = 4
            #pdb.set_trace()
            #print(thr[0])
            #print(np.mean(arr))
    return thr

def spike_detection(recording_traces,params):
    #######################SINGLE CHANNEL###############################################
    if (len(recording_traces.shape) ==1):
        #emphasizer 
        recording_emphasized = emphasizer(recording_traces, params)
        #threshold
        thr = thr_comp(recording_emphasized,params)
        #thr = thr_comp(np.abs(recording_emphasized),params) # the wrong DVT settings ---
        # detection strategy
        spike_instants = detection_1D(recording_emphasized, thr,params)



        plot_on = 0
        if plot_on == 1:
            plt.figure(figsize=(10, 6))
            # 绘制 recording_emphasized 的线形图
            plt.plot(recording_emphasized[1:30000], label='Recording Emphasized', marker='o')
            plt.plot(recording_traces[1:30000], label='Recording raw', marker='*')
            # 在阈值 thr 处添加水平线
            plt.axhline(y=thr, color='r', linestyle='--', label='Threshold')
            plt.axhline(y=-thr, color='r', linestyle='--', label='Threshold')
            # 添加标签和标题
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.title('Visualization of Recording Emphasized with Threshold')

            # 显示图例
            plt.legend()

            # 显示图形
            plt.savefig('detection.jpg')
            plt.show()
        spike_coords = np.zeros_like(spike_instants,int)

    else:
        #emphasizer (30000,128)
        #threshold (30000,128) --online
        multi_align_en = params['multi_align_en']
        recording_emphasized = np.zeros_like(recording_traces)
        thr = np.zeros_like(recording_traces)
        for i in range(recording_traces.shape[1]):
            recording_emphasized[:,i] = emphasizer(recording_traces[:,i], params)
        
        for i in range(recording_traces.shape[1]):
            thr[:,i] = thr_comp(recording_emphasized[:,i],params)       
        # detection strategy
        spike_not_aligned_instants,spike_coords = LC_spatio_detection(recording_emphasized, thr,params)
        if(multi_align_en == True):
            spike_instants,spike_coords = LC_spatio_alignment(params,recording_emphasized,spike_not_aligned_instants,spike_coords,LC_window=1,stride=1)
        else:
            spike_instants = spike_not_aligned_instants
    return spike_instants, spike_coords,recording_emphasized,thr

def one_d_convolution(input_array, kernel_size,stride):
    # 定义卷积核，这里使用全1的1D矩阵
    kernel = np.ones(kernel_size)
    # 执行一维卷积操作
    result = np.convolve(input_array, kernel, mode='same')
    #pdb.set_trace()
    return result[int(kernel_size/2)::stride]

def LC_alignment(params,recording_traces,spike_not_aligned_instants,LC_window,stride,coords = None):
    align_mode = params['align_mode']
    align_canceling_window = params['align_canceling_window'][1]
    spike_length = np.array([0,align_canceling_window])  #wrongly *int(LC_window/stride)
    instants_aligned = []
    channel_count = recording_traces.shape[1] if len(recording_traces.shape) > 1 else 1
    if channel_count==1:
        for instant in spike_not_aligned_instants:
            start = instant+spike_length[0]
            end = instant+spike_length[1]
            window_data = recording_traces[start:end] # windowing
            if(align_mode == 1):  #max slope
                index = np.argmax(window_data)+start # indexing no diff as LC data is already diffed
            elif(align_mode == 0): #max_abs
                reconstructed_data = np.cumsum(window_data)
                index = np.argmax(np.abs(reconstructed_data))+start # indexing
            instants_aligned.append(index) 
            #pdb.set_trace()
        spike_instants = np.array(instants_aligned)
    else: 
        for i, instant in enumerate(spike_not_aligned_instants):
            start = instant+spike_length[0]
            end = instant+spike_length[1]
            window_data = recording_traces[start:end,coords[i]] # windowing
            index = np.argmax(window_data)+start # indexing no diff as LC data is already diffed
            #index = np.argmax(np.abs(window_data))+instant # indexing
            instants_aligned.append(index) 
            #pdb.set_trace()
        spike_instants = np.array(instants_aligned)
    sorted_indices = np.argsort(spike_instants)

    # 使用排序索引重新排列数组
    sorted_a_spike_instants = spike_instants[sorted_indices]
    #sorted_a_spike_coords = spike_coords[sorted_indices]
    return sorted_a_spike_instants

def LC_spatio_alignment(params,recording_traces,spike_not_aligned_instants,spike_coords,LC_window,stride):
    channel_count = params['channel_count']
    align_canceling_window = params['align_canceling_window']
    a_wind_lines = align_canceling_window[0] #2*(6*2+1) = 26个neighbor
    a_wind_points = align_canceling_window[1] # 之前是20

    spike_length = np.array([0,a_wind_points])*int(LC_window/stride) #???不应该是up_times嘛 #alignment_length
    # 处理每个脉冲实例
    n_coords = recording_traces.shape[1]
    a_spike_instants = np.zeros_like(spike_not_aligned_instants)
    a_spike_coords = np.zeros_like(spike_not_aligned_instants)
    align_mode = params['align_mode']
    for i, instant in enumerate(spike_not_aligned_instants):
    

        c = spike_coords[i] #当前coord
        l_c = int(max(0, c - 2 * a_wind_lines - (c ) % 2))
        r_c = int(min(channel_count-1, c + 2 * a_wind_lines + (c+1) % 2)) #对齐的coord范围
        start = max(0, instant +spike_length[0] )
        end = min(recording_traces.shape[0], instant +spike_length[1] )
        ref = - np.inf
        # 在脉冲窗口中找到最大slope
        for s in range(l_c, r_c+1):
            spike_s = recording_traces[start:end, s]
            spike_reconstructed = np.cumsum(spike_s)
            if (align_mode == 'max_slope_LC'):
                tmp_ref = np.max(spike_s) #tmp_ref = np.max(spike_s) #max slope alignment for LC ADC, recording traces is already diffed
            elif (align_mode == 'max_abs_LC'): 
                tmp_ref = np.max(np.abs(spike_reconstructed))
            elif (align_mode == 'max_abs_tra'): 
                tmp_ref = np.max(np.abs(spike_s)) 
            elif (align_mode == 'max_slope_tra'): 
                tmp_ref = np.max(np.diff(spike_s)) 
            if tmp_ref > ref:
                ref = tmp_ref
                if (align_mode == 'max_slope_LC'):
                    inst = np.argmax(spike_s)+ start 
                elif (align_mode == 'max_abs_LC'): 
                    inst = np.argmax(np.abs(spike_reconstructed))+ start
                elif (align_mode == 'max_abs_tra'): 
                    inst = np.argmax(np.abs(spike_s))+ start 
                elif (align_mode == 'max_slope_tra'): 
                    inst = np.argmax(np.diff(spike_s))+ start 
                coord = s
        a_spike_instants[i] = inst
        a_spike_coords[i] = coord  

        # 解压排序后的数组
        sorted_indices = np.argsort(a_spike_instants)

        # 使用排序索引重新排列数组
        sorted_a_spike_instants = a_spike_instants[sorted_indices]
        sorted_a_spike_coords = a_spike_coords[sorted_indices]
        #start0 = max(1, inst +spike_length[0] - 1)
        #end0 = min(recording_traces.shape[0], inst +spike_length[1] - 1)   
        #spike_reconstructed = np.cumsum(recording_traces[start0:end0, coord])     
        #plt.plot(spike_reconstructed)     
        #plt.show()
    #return a_spike_instants, a_spike_coords
    return sorted_a_spike_instants, sorted_a_spike_coords



def LC_detect(pulse_train,LC_window, params, stride=None,abs_mode=True,thr_predefined = None):
    if abs_mode == True:
        used_pulse_train = np.abs(pulse_train)
    else:
        used_pulse_train = pulse_train
    if stride== None:
        stride = LC_window

    channel_count = pulse_train.shape[1] if len(pulse_train.shape) > 1 else 1
    LC_cnt_arr = np.zeros((int(used_pulse_train.shape[0]/stride),channel_count),dtype = np.int16)
    if (pulse_train.ndim == 1):
        pdb.set_trace()
        LC_cnt_arr = one_d_convolution(used_pulse_train,LC_window,stride)
    else:
        for i in range(channel_count):
            LC_cnt_arr[:,i] = one_d_convolution(used_pulse_train[:,i],LC_window,stride)

    #thr_comp and detect and align
    if (channel_count == 1):
        LC_thr=thr_comp(np.abs(pulse_train),params,LC_window,thr_predefined)[stride-1::stride]
        print(np.unique(LC_thr))
        spike_instants= detection_1D(LC_cnt_arr,LC_thr,params,stride)
        #spike_instants_highfreq = detection(LC_cnt_arr,np.ones_like(LC_cnt_arr)*LC_thr,params,stride)
        spike_instants_highfreq_aligned = None
        spike_coords = None
    else:
        LC_thr = np.zeros((int(used_pulse_train.shape[0]/stride),channel_count))
        for i in range(channel_count):
            LC_thr[:,i] = thr_comp(np.abs(pulse_train[:,i]),params,LC_window)[stride-1::stride]
            
            #LC_thr[:,i] = thr_comp(LC_cnt_arr[:,i],params,LC_window)
        spike_instants_highfreq,d_spike_coords = LC_spatio_detection(LC_cnt_arr, LC_thr,params)
        
        #spike_instants_highfreq_aligned = LC_alignment(LC_cnt_arr,spike_instants_highfreq,LC_window,stride,d_spike_coords)
        #spike_instants = np.round(spike_instants_highfreq_aligned*stride/up_times).astype(int)
        #spike_coords = d_spike_coords
        if(params['multi_align_en'] == 1):
            spike_instants,spike_coords = LC_spatio_alignment(params,LC_cnt_arr,spike_instants_highfreq,d_spike_coords,LC_window,stride)
        else:
            spike_instants = spike_instants_highfreq
            spike_coords = d_spike_coords
        spike_instants_highfreq_aligned = None
    return spike_instants,spike_instants_highfreq_aligned,LC_cnt_arr,spike_coords,LC_thr


def snippets_LC_cnt(LC_pt_spike_snippets,LC_pt_spike_neigh_snippets, LC_params):
    stride = LC_params['stride']
    LC_window = LC_params['LC_window']
    LC_cnt_spike_snippets = np.zeros((int(LC_pt_spike_snippets.shape[0]),int(LC_pt_spike_snippets.shape[1]/stride)),dtype = np.int16)

    for i, item in enumerate(LC_pt_spike_snippets):
        #pdb.set_trace()
        LC_cnt_spike_snippets[i,:] = one_d_convolution(item,LC_window,stride)


    LC_cnt_spike_neigh_snippets = None
    
    if LC_pt_spike_neigh_snippets.ndim ==3:
        LC_cnt_spike_neigh_snippets =np.zeros((LC_pt_spike_neigh_snippets.shape[0],int(LC_pt_spike_neigh_snippets.shape[1]/stride), LC_pt_spike_neigh_snippets.shape[2]),dtype = np.int8)
        for i, item in enumerate(LC_pt_spike_neigh_snippets):
            for j in range(LC_pt_spike_neigh_snippets.shape[2]):
                LC_cnt_spike_neigh_snippets[i,:,j] = one_d_convolution(item[:,j],LC_window,stride)
    return LC_cnt_spike_snippets,LC_cnt_spike_neigh_snippets

def LC_cnt_only(pulse_train, LC_window, stride=None,abs_mode=False):
    if abs_mode == True:
        used_pulse_train = np.abs(pulse_train)
    else:
        used_pulse_train = pulse_train
    if stride== None:
        stride = LC_window

    channel_count = pulse_train.shape[1] if len(pulse_train.shape) > 1 else 1
    LC_cnt_arr = np.zeros((int(used_pulse_train.shape[0]/stride),channel_count),dtype = np.int16)
    if (pulse_train.ndim == 1):
        LC_cnt_arr = one_d_convolution(used_pulse_train,LC_window,stride)
    else:
        for i in range(channel_count):
            LC_cnt_arr[:,i] = one_d_convolution(used_pulse_train[:,i],LC_window,stride)
    return LC_cnt_arr