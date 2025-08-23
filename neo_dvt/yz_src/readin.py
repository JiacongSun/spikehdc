#[fs, coords, gt_instants,gt_labels] = readin(recording_path,gt_path, length_of_recording,data_type)
# coords: x/y information
# recording_traces: length*coords, numpy, raw recording; index follows the same as Bernardo's (x-x=y-y...)
# sortingGT: sorting objects, with instants(time) and labels (ID)
import numpy as np
import spikeinterface.extractors as se
import MEArec as mr
import scipy.io as sio
import ast
import matplotlib.pyplot as plt
from scipy import signal
import pdb
from scipy.io import loadmat
def readin (recording_path,gt_path, length_of_recording,data_type,set_name = None):
    if data_type == "Quiroga":
        quiroga_dir = "/imec/other/macaw/projectdata/quiroga_datasets/"
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
                     f"{quiroga_dir}C_Difficult2_noise02.mat"
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
        recording_object = se.NumpyRecording(traces_list=[recording_traces.reshape(-1,1)], sampling_frequency=fs)
        from probeinterface import Probe
        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=np.array([0,0]).reshape(1,-1))
        probe.set_device_channel_indices(np.arange(1))
        recording_object.set_probe(probe,in_place = True)
        '''
        gt_instants_aligned = []
        window_size = 9
        for instant in gt_instants:
            start = instant
            end = instant + window_size
            window_data = recording_traces[start:end] # windowing
            index = np.argmax(np.abs(window_data))+instant # indexing
            gt_instants_aligned.append(index) 
        gt_instants = gt_instants_aligned
        '''
    elif data_type == "UCLA":
        sortingGT_all = se.read_waveclus(file_path = '/imec/other/macaw/projectdata/UCLA_data/UCLA_data/times_CSC4.mat')
        recording = se.read_neuralynx(folder_path = '/imec/other/macaw/projectdata/UCLA_data/UCLA_data')
        fs = recording.get_sampling_frequency()
        if (length_of_recording>1903):
            recording_traces = recording.get_traces()
        else:
            recording_traces = recording.get_traces(end_frame=int(length_of_recording*fs))
        spike_vector_all = sortingGT_all.to_spike_vector()
        selected_rows = [row for row in spike_vector_all if row[0] < int(length_of_recording*fs)] # select the spikes before the end frame
        gt_instants = [row[0] for row in selected_rows]
        gt_labels = [row[1] for row in selected_rows]
        new_coords = [0,0]
        gt_c_coords = np.zeros([len(gt_instants)],int)
        recording_traces = recording_traces.squeeze()
        centering_positions = None
    elif data_type == 'YP': #Yingping's datasets
        
        
        if(set_name == '1'):
            centering_new_idx = [ 75,  70,  78,  28, 111,  92,  90]
            record_path = '/imec/other/macaw/projectdata/neuropixels_datasets/set_yingping1/20141202_all_es.xml'
            file_path = '/imec/other/macaw/projectdata/neuropixels_datasets/set_yingping1/imecToWhisper_dict.prb'
            mat_gtdata = sio.loadmat('/imec/other/macaw/projectdata/neuropixels_datasets/set_yingping1/20141202_all_es_gtTimes')
        elif(set_name == '2'):
            centering_new_idx = [103, 87, 87, 90, 83, 91, 106]
            record_path = '/imec/other/macaw/projectdata/neuropixels_datasets/set_yingping2/20150924_1_e.xml'
            file_path = '/imec/other/macaw/projectdata/neuropixels_datasets/set_yingping2/imecToWhisper_20150924_dict.prb'
            mat_gtdata = sio.loadmat('/imec/other/macaw/projectdata/neuropixels_datasets/set_yingping2/20150924_1_e_gtTimes')
        elif(set_name == '3'):
            centering_new_idx = [54, 85, 38, 102, 74, 76, 78, 119]  #Ber叔 [55, 87, 40, 104, 75, 77, 79, 120]
            record_path = '/imec/other/macaw/projectdata/neuropixels_datasets/set_yingping3/20150601_all_s.xml'
            file_path = '/imec/other/macaw/projectdata/neuropixels_datasets/set_yingping3/imecToWhisper_dict.prb'
            mat_gtdata = sio.loadmat('/imec/other/macaw/projectdata/neuropixels_datasets/set_yingping3/20150601_all_s_gtTimes')

        # recording readin
        recording = se.read_neuroscope(file_path=record_path)
        fs = recording.get_sampling_frequency()
        recording_traces = recording.get_traces(end_frame=int(length_of_recording*fs))
        
        ########################################recording reordering #####################################
        

        with open(file_path, 'r') as file:
            file_content = file.read()
        # 使用 ast.literal_eval 将文件内容转换为字典
        probe_data = ast.literal_eval(file_content)
        channels = probe_data[1]['channels']
        geometry_dict = probe_data[1]['geometry']
        ori_coords = np.array([geometry_dict[channel] for channel in channels]) #得出原始channel对应的coords  

        


        # 生成ideal_connected和unconnected的channel信息     
        ideal_connected = np.zeros([130,2],int) 
        for i in range(65):
            for j in range(2):
                ideal_connected[2*i+j,:] = [j*20,i*20]      #这个ideal connected才是最后的coords!!!
        unconnected = []
        for i in range(128):
            target_row = ideal_connected[i]
            indices = np.where((ori_coords[:, 0] == target_row[0]) & (ori_coords[:, 1] == target_row[1]))
            if(len(indices[0]) == 0):
                unconnected.append(i) 
  
        recording_traces_arr = np.array(recording_traces)
        recording_reordered = np.zeros_like(recording_traces_arr)
        index_in_channels = 0
        mapped_rule = [] #原始的（120，）channels 和reorder之后的对应关系。
        for i in range(128):
            if (i in unconnected):
                continue
            else:
                #print(index_in_channels)
                recording_reordered[:,i] = recording_traces_arr[:,channels[index_in_channels]]
                #print(recording_reordered[:,i] )
                index_in_channels=index_in_channels+1
                mapped_rule.append(i)
        recording_traces = recording_reordered #完成reording of recording
        #################################删除最后一行!!!!!!!!!!!!!!!!!!!!!!#################################
        recording_traces =recording_traces[:,0:128]

        new_coords = ideal_connected #传出的是改变后的coords
        ## GT information
        

        gt_times = mat_gtdata['gtTimes'][0] # 提取gt_times
        num_events = len(gt_times) # 获取事件数量
        gt = np.zeros((2, 0)) # 初始化结果数组gt

        for event_id in range(1, num_events + 1): # 遍历每个事件
            event_times = np.array(gt_times[event_id - 1]).ravel() # 获取当前事件的时间数组
            event_data = np.array([event_id * np.ones_like(event_times), event_times]) # 将事件ID和时间数组转换为[2, n]数组
            gt = np.concatenate((gt, event_data), axis=1).astype(int) # 将当前事件数据添加到gt数组中
        sorted_indices = np.argsort(gt[1, :]) # 使用 argsort 对第二行(time)进行排序，返回排序后的索引
        sorted_gt = gt[:, sorted_indices] # 使用排序后的索引重新排列整个数组
        gt_labels_all = sorted_gt[0]
        gt_instants_all = sorted_gt[1]     #total labels and instants
        selected_index = np.where(gt_instants_all<int(length_of_recording*fs))
        gt_labels = gt_labels_all[selected_index]
        gt_instants = gt_instants_all[selected_index]
        # new coords(传出了新coords),  gt_c_coords,centering_positions(x_y坐标)

        ########################产生centering channels ####################################
        ori_idx_group = [] #原始index数组
        for i in range(num_events):
            ori_idx_group.append(mat_gtdata['gtChans'][0][i].ravel())
        new_idx_group = []
        for m in range(num_events): #reorder后的index数组
            current_idx_group = []
            for i in ori_idx_group[m]:
                if i in channels:
                    ori_index = channels.index(i)
                    new_idx = mapped_rule[ori_index]
                    current_idx_group.append(new_idx)
            new_idx_group.append(current_idx_group)
        
        
        '''
        centering_new_idx = []
        for m in range(num_events):
            first_event_time = gt_times[m][0][0]
            current_group = new_idx_group[m]
            current_snippets = recording_reordered[first_event_time-50:first_event_time+50,current_group]
            max_index = np.argmax(abs(current_snippets))
            max_index_2d = np.unravel_index(max_index, current_snippets.shape) # 将一维索引转换为二维索引
            centering_new_idx.append(current_group[max_index_2d[1]])
        '''
        #centering_new_idx = [55, 87, 40, 104, 75, 77, 79, 120]
        centering_new_idx = np.array(centering_new_idx)  # [ 75,  70,  78,  28, 111,  92,  90]
        gt_c_coords = centering_new_idx[(gt_labels-1).tolist()].tolist() #对应gt_labels,一个长为和spike数量相同的数组，每个元素是该spike的new_coords
        centering_positions = ideal_connected[gt_c_coords].tolist() #对应gt_labels,一个长为和spike数量相同的数组，每个元素是该spike的物理坐标

        #生成probe和recording object
        import spikeinterface as si
        recording_object = si.NumpyRecording(recording_traces,fs)
        from probeinterface import Probe
        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=new_coords[0:128])
        probe.set_device_channel_indices(np.arange(128))
        recording_object.set_probe(probe,in_place = True)

    elif data_type == "MEArec":
        #print('read mearec')
        recording,sortingGT_all = se.read_mearec(recording_path)
        recording_object = recording
        fs = recording.get_sampling_frequency()
        recording_traces = recording.get_traces(end_frame=int(length_of_recording*fs))
        spike_vector_all = sortingGT_all.to_spike_vector()
        selected_rows = [row for row in spike_vector_all if row[0] < int(length_of_recording*fs)] # select the spikes before the end frame
        gt_instants = [row[0] for row in selected_rows]
        gt_labels = [row[1] for row in selected_rows]
        ## xy_coords, chanmap and reordered recording
        if recording_traces.shape[1] != 1: #multi channel cases, 这里头的coords是ori_coords
            probe = recording.get_probe()
            df = probe.to_dataframe()
            coords = df.iloc[:, :2].values #x_ycoords
            chanmap = np.lexsort((coords[:, 0], coords[:, 1])) # sort by y coords, then by x
            recording_traces = recording_traces[:,chanmap]
            ## gt_coords and gt_c_coords
            recgen = mr.load_recordings(recording_path)
            #spike_templates = np.array(recgen.templates) #60,10,128,292
            #pdb.set_trace()
            #mr.plot_templates(recgen, template_ids=[0,1,2,3])
            voltage_peaks = np.array(recgen.voltage_peaks)
            max_channel = np.argmax(np.abs(voltage_peaks), axis=1) #原始索引，（160，），跟coords对应
            indices = np.zeros(len(max_channel)) 
            for i in range(len(max_channel)):
                indices[i] = np.where(np.array(chanmap) == max_channel[i])[0]
            c_coords = indices.astype(int) #更改后的索引，（160，），按照行排列
            # c_coords = chanmap[max_channel] # wrong centering coords
            gt_c_coords = c_coords[gt_labels].tolist()
            gt_c_channel = max_channel[gt_labels].tolist()
            centering_positions = coords[gt_c_channel].tolist()
            new_coords = coords[chanmap]
        else: # single channel cases
            new_coords = [0,0]
            gt_c_coords = np.zeros([len(gt_instants)],int)
            recording_traces = recording_traces.squeeze()
            centering_positions = None
    elif data_type =='NP':
        #现在只支持set2

        folder = "/imec/other/macaw/projectdata/neuropixels_datasets/set"+set_name+"/"
        data1 = loadmat(folder+"curated_ST/set"+set_name+"_ground_truth.mat")
        data2 = loadmat(folder+"curated_ST/set"+set_name+"_clusters_matrix_single_ok.mat")
        recording = se.read_spikeglx(folder_path=folder)
        recording_object = recording
        #暂时在这里加入preprocessing
        from spikeinterface.preprocessing import bandpass_filter, common_reference
        #rec_f = bandpass_filter(recording, freq_min=300, freq_max=6000)
        recording = common_reference(recording, reference="global",operator="median")

        #fs = recording.get_sampling_frequency()
        fs = 30000
        recording_traces = recording.get_traces(end_frame=int(length_of_recording*fs))
        
        probe = recording.get_probe()
        df = probe.to_dataframe()
        coords = df.iloc[:, :2].values #x_ycoords
        chanmap = np.lexsort((coords[:, 0], coords[:, 1])) # sort by y coords, then by x # in np4, it's totally in order
        new_coords = coords[chanmap]
        recording_traces = recording_traces[:,chanmap]

        ###########gt_instants, gt_labels, gt_c_coords, centering positions
        ground_truth = data1['ground_truth'].astype(int)
        clusters_matrix = data2['clusters_matrix'].astype(int)
        
        selected_rows = [row for row in ground_truth if row[0] < int(length_of_recording*fs)] # select the spikes before the end frame
        gt_all_instants = [row[0] for row in selected_rows]
        gt_all_labels = [row[1] for row in selected_rows]

        gt_c_coords = []
        gt_instants = []
        gt_ori_labels = []
        gt_new_labels = []

        for i, label in enumerate(gt_all_labels):
            
            coord = clusters_matrix[np.where(clusters_matrix[:,1]==label),4][0]
            if(len(coord)>0):
                index = np.where(clusters_matrix[:,1]==label)[0][0]
                gt_instants.append(gt_all_instants[i])
                gt_ori_labels.append(gt_all_labels[i])
                gt_new_labels.append(clusters_matrix[index,0]+1)
                gt_c_coords.append(coord[0] -1)
            #else:
            #    print('no')
        centering_positions = new_coords[gt_c_coords].tolist()
        gt_labels = gt_new_labels







        
    return fs, new_coords, recording_traces, gt_instants,gt_labels,gt_c_coords,centering_positions,recording_object



def data_plotting(signal_data,fs,a=0.1):

    time = np.linspace(0, a, int(fs * a), endpoint=False)
    #frequencies, spectrum = signal.welch(signal_data, fs, nperseg=256)
    # 计算信号的傅里叶变换
    fft_result = np.fft.fft(signal_data)
    fft_freq = np.fft.fftfreq(len(fft_result), 1/fs)
    # 获取频谱的幅度谱
    magnitude_spectrum = np.abs(fft_result)
    # 计算信号的 spectrogram
    #f, t, spectrogram = signal.spectrogram(signal_data[:int(fs*a)], fs, nperseg=256)

    plt.figure(figsize=(12, 6))
    # 绘制频谱图
    plt.subplot(2, 1, 1)
    plt.plot(fft_freq, 10*np.log10(magnitude_spectrum))
    plt.title('spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.xlim([0,fs/2])  # 设置 x 轴范围
    #plt.ylim([0,2000])  # 设置 y 轴范围
    # 绘制 spectrogram
    #plt.subplot(3, 1, 2)
    #plt.pcolormesh(t, f, 10 * np.log10(spectrogram), shading='auto')
    #plt.title('Spectrogram')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Freq (Hz)')
    # 绘制 transient
    plt.subplot(2, 1, 2)
    plt.plot(time,signal_data[:int(fs*a)])
    plt.title('raw data')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim([0, a])  # 设置 x 轴范围
    plt.ylim([-1.5, 1.5])  # 设置 x 轴范围
    plt.tight_layout()
    plt.show()

    return None