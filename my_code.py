#!/usr/bin/env python
# coding: utf-8

# In[1]:


def load_all_datasets(path):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import mne
    from mne.io import read_raw_edf
    list_files = os.listdir(path=path)
    
    extension = '.edf'
    index = 0
    list_dataset = []
    for file in list_files:
        if extension in list_files[index]:
            list_dataset.append(list_files[index])
        index += 1

    list_load_dataset = []
    for n_file in range(0, len(list_dataset)):
        dataset = read_raw_edf(list_dataset[n_file], preload=True)
        list_load_dataset.append(dataset)
        
    return list_load_dataset


# In[3]:


# dont repeat for datasets ......; do it once 

def preliminary_steps(raw_datasets):
    import mne
    pre_processed_datasets = []
    
    # re-referencing the data to 'CQ_CMS', 'CQ_DRL'
    
    for dataset in raw_datasets:
        dataset.set_eeg_reference(ref_channels=['CQ_CMS', 'CQ_DRL'])
    
    # selecting only the electrodes of interest
    
        include_channels = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
        
        for channels in dataset.ch_names:
            if channels not in include_channels:
                dataset.drop_channels(channels)
            
    # setting montage 
    
        dataset.set_montage(mne.channels.make_standard_montage("standard_1020"))
        
    
    # high pass filter to remove slow drifts ---> built in EMOTIV??
    
        dataset = dataset.copy().filter(l_freq=0.16, h_freq=None)

        
    # notch filter to remove powerline noise  ---> built in EMOTIV?? 
    
        #freqs = (50, 100)
        #dataset = dataset.copy().notch_filter(freqs=freqs)
        
    
    # annotating raw data
    
        # we can create multiple annotations at the same time
        # total time -> 60s + 5 x(20s) = 160 s

        start_time = 60
        delay = 1
        dur = 20
        len_event = 7 
        tot_time = start_time + delay 


        start_push = [tot_time + 0*dur, tot_time + 1*dur, tot_time+2*dur, tot_time+3*dur, tot_time+4*dur]
        start_relax = [tot_time + 0*dur+10, tot_time + 1*dur+10, tot_time+2*dur+10, tot_time+3*dur+10, tot_time+4*dur+10]

        
        push_annotations = mne.Annotations(onset=start_push, duration=[len_event]*5, description=["Push"]*5, orig_time=dataset.info['meas_date'])
        relax_annotations = mne.Annotations (onset=start_relax, duration=[len_event]*5, description=["Relax"]*5, orig_time=dataset.info['meas_date'])

        dataset.set_annotations(push_annotations+relax_annotations)
        
    
    # creating events from annotations 
    
    
    
    # now that we have annotations, we will tranfer them into events
    # this is needed to be able to then create epochs
        events_from_annot, event_dict = mne.events_from_annotations(dataset)
        pre_processed_datasets.append(dataset)
    return pre_processed_datasets
        


# In[ ]:


def create_epochs(pre_processed_datasets):
    import mne
    events_from_annot, event_dict = mne.events_from_annotations(pre_processed_datasets[0])
    delay = 0.5
    
    baseline = (0.5, 0.5)
    event_dict = {"Push" : 1, "Relax" : 2}
    # not sure what to set it to; resting state activity looks super noisy 
    epochs_all = mne.Epochs(pre_processed_datasets[0], events=events_from_annot, event_id = event_dict, baseline = baseline, tmin = 0.5, tmax = (10-delay), preload = True, reject_by_annotation=False)
    # reject by annotation argument passed
    
    for dataset in pre_processed_datasets:
        
        baseline = (0.5, 0.5)
        epochs = mne.Epochs(dataset, events=events_from_annot, event_id = event_dict, baseline = baseline, tmin = 0.5, tmax = (10-delay), preload = True, reject_by_annotation=False)
        epochs_all = mne.concatenate_epochs([epochs_all, epochs])
        
    epochs_all.drop([0,1,2,3,4,5,6,7,8,9])
    return epochs_all
    


# In[ ]:


def clean_epochs(epochs_all):
        
    import autoreject
    import numpy as np 
    from autoreject import AutoReject
    from autoreject import get_rejection_threshold
    import mne
    
    # creating a random list of parameters which will be modified by learning 
    # n_interpolates are the ρ values that we would like autoreject to try
    # consensus_percs are the κ values that autoreject will try 
    
    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)
    
    # specifying the channel type
    info = epochs_all.info
    picks = mne.pick_types(info, meg=False, eeg=True, stim=False, eog=False, ecg=False, emg=False, ref_meg='auto', misc=False, resp=False, chpi=False, exci=False, ias=False, syst=False, seeg=False, dipole=False, gof=False, bio=False, ecog=False, fnirs=False, include=(), exclude='bads', selection=None)
    
    # initiating the algorithm 
    
    ar = AutoReject(n_interpolates, consensus_percs, picks=picks,
                thresh_method='random_search', random_state=42)

    
    # need to fit the ar first in order to then transform the epochs which can be repaired 
    
    ar.fit(epochs_all)
    
    print("Rejection thresholds per channel/n:")
    for ch_name in epochs_all.info['ch_names']:
         print('%s: %s' % (ch_name, ar.threshes_[ch_name]))
    
    
    reject_log = ar.get_reject_log(epochs_all)
    reject_log.plot()
    reject_log.plot_epochs(epochs_all)
    
    # repairing epochs where possible 
    
    cleaned_epochs = ar.transform(epochs_all)

    
    return cleaned_epochs
    
    


# In[ ]:


def create_epochs_rejections(pre_processed_datasets):
    import mne 
    reject_criteria = dict(eeg=150e-6) 
    events_from_annot, event_dict = mne.events_from_annotations(pre_processed_datasets[0])
    delay = 1
    
    baseline = (1, 1)
    event_dict = {"Push" : 1, "Relax" : 2}
    # not sure what to set it to; resting state activity looks super noisy 
    epochs_all_rejected = mne.Epochs(pre_processed_datasets[0], events=events_from_annot, event_id = event_dict, baseline = baseline, tmin = 0.5, tmax = (9-delay), preload = True, reject_by_annotation=False)
    # reject by annotation argument passed
    
    for dataset in raw_datasets:
        
        baseline = (1, 1)
        epochs = mne.Epochs(dataset, events=events_from_annot, event_id = event_dict, baseline = baseline, tmin = 0.5, tmax = (9-delay), preload = True, reject_by_annotation=False, reject=reject_criteria, reject_tmin=0.5, reject_tmax=3)
        epochs_all_rejected = mne.concatenate_epochs([epochs_all_rejected, epochs])
        
    epochs_all_rejected.drop([0,1,2,3,4,5,6,7,8,9])
    return epochs_all_rejected
    


# In[ ]:


def apply_ica(cleaned_epochs):
    import numpy as np
    from mne.preprocessing import ICA
    ica = ICA(n_components=14, random_state=97)
    ica.fit(cleaned_epochs)

    ica.plot_sources(cleaned_epochs, show_scrollbars=False)
    ica.plot_components()
    
    components = np.arange(0,14)
    for component in components:
        ica.plot_properties(cleaned_epochs, picks=component)


# In[ ]:


def extract_freqs(low_band, high_band, num):
    import numpy as np
    freqs = np.logspace(*np.log10([low_band, high_band]), num=num)

    return freqs


# In[6]:


def plot_topomaps(alpha_push_power, alpha_relax_power,  beta_push_power, beta_relax_power,  gamma_push_power,  gamma_relax_power):
    
    baseline=(0.5,0.5)

    alpha_push_power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=9.5, fmin=8, fmax=12,
                   baseline=baseline,
                   title='Push Alpha', show=False, contours=1)
    alpha_relax_power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=9.5, fmin=8, fmax=12,
                   baseline=baseline,
                   title='Relax Alpha', show=False, contours=1)

    beta_push_power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=9.5, fmin=12, fmax=30,
                   baseline=baseline,
                   title='Push Beta', show=False, contours=1)
    beta_relax_power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=9.5, fmin=12, fmax=30,
                   baseline=baseline,
                   title='Relax Beta', show=False, contours=1)

    gamma_push_power.plot_topomap(ch_type='eeg',tmin=0.5, tmax=9.5, fmin=30, fmax=80,
                   baseline=baseline,
                   title='Push Gamma', show=False, contours=1)
    gamma_relax_power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=9.5, fmin=30, fmax=80,
                   baseline=baseline,
                   title='Relax Gamma', show=False, contours=1)


# In[2]:


# to fix slicing whole timepoints 

def visualize_per_channel_power_diffs(epochs_all, alpha_push_power, alpha_relax_power,  beta_push_power, beta_relax_power,  gamma_push_power,  gamma_relax_power):

    import numpy as np
    import matplotlib.pyplot as plt
    
    power_push_alpha_data = alpha_push_power.data
    power_push_alpha_data_average = power_push_alpha_data.mean(axis=1)
    power_relax_alpha_data = alpha_relax_power.data
    power_relax_alpha_data_average = power_relax_alpha_data.mean(axis=1)

    difference_AF3_alpha = power_push_alpha_data_average[0] - power_relax_alpha_data_average[0]
    difference_AF4_alpha = power_push_alpha_data_average[13] - power_relax_alpha_data_average[13]
    difference_F7_alpha = power_push_alpha_data_average[1] - power_relax_alpha_data_average[1]
    difference_F8_alpha = power_push_alpha_data_average[12] - power_relax_alpha_data_average[12]
    difference_F3_alpha = power_push_alpha_data_average[2] - power_relax_alpha_data_average[2]
    difference_F4_alpha = power_push_alpha_data_average[11] - power_relax_alpha_data_average[11]
    difference_FC5_alpha = power_push_alpha_data_average[3] - power_relax_alpha_data_average[3]
    difference_FC6_alpha = power_push_alpha_data_average[10] - power_relax_alpha_data_average[10]
    difference_T7_alpha = power_push_alpha_data_average[4] - power_relax_alpha_data_average[4]
    difference_T8_alpha = power_push_alpha_data_average[9] - power_relax_alpha_data_average[9]
    difference_P7_alpha = power_push_alpha_data_average[5] - power_relax_alpha_data_average[5]
    difference_P8_alpha = power_push_alpha_data_average[8] - power_relax_alpha_data_average[8]
    difference_O1_alpha = power_push_alpha_data_average[6] - power_relax_alpha_data_average[6]
    difference_O2_alpha = power_push_alpha_data_average[7] - power_relax_alpha_data_average[7]

    power_push_beta_data = beta_push_power.data
    power_push_beta_data_average = power_push_beta_data.mean(axis=1)
    power_relax_beta_data = beta_relax_power.data
    power_relax_beta_data_average = power_relax_beta_data.mean(axis=1)

    difference_AF3_beta = power_push_beta_data_average[0] - power_relax_beta_data_average[0]
    difference_AF4_beta = power_push_beta_data_average[13] - power_relax_beta_data_average[13]
    difference_F7_beta = power_push_beta_data_average[1] - power_relax_beta_data_average[1]
    difference_F8_beta = power_push_beta_data_average[12] - power_relax_beta_data_average[12]
    difference_F3_beta = power_push_beta_data_average[2] - power_relax_beta_data_average[2]
    difference_F4_beta = power_push_beta_data_average[11] - power_relax_beta_data_average[11]
    difference_FC5_beta = power_push_beta_data_average[3] - power_relax_beta_data_average[3]
    difference_FC6_beta = power_push_beta_data_average[10] - power_relax_beta_data_average[10]
    difference_T7_beta = power_push_beta_data_average[4] - power_relax_beta_data_average[4]
    difference_T8_beta = power_push_beta_data_average[9] - power_relax_beta_data_average[9]
    difference_P7_beta = power_push_beta_data_average[5] - power_relax_beta_data_average[5]
    difference_P8_beta = power_push_beta_data_average[8] - power_relax_beta_data_average[8]
    difference_O1_beta = power_push_beta_data_average[6] - power_relax_beta_data_average[6]
    difference_O2_beta = power_push_beta_data_average[7] - power_relax_beta_data_average[7]

    power_push_gamma_data = gamma_push_power.data
    power_push_gamma_data_average = power_push_gamma_data.mean(axis=1)
    power_relax_gamma_data = gamma_relax_power.data
    power_relax_gamma_data_average = power_relax_gamma_data.mean(axis=1)

    difference_AF3_gamma = power_push_gamma_data_average[0] - power_relax_gamma_data_average[0]
    difference_AF4_gamma = power_push_gamma_data_average[13] - power_relax_gamma_data_average[13]
    difference_F7_gamma = power_push_gamma_data_average[1] - power_relax_gamma_data_average[1]
    difference_F8_gamma = power_push_gamma_data_average[12] - power_relax_gamma_data_average[12]
    difference_F3_gamma = power_push_gamma_data_average[2] - power_relax_gamma_data_average[2]
    difference_F4_gamma = power_push_gamma_data_average[11] - power_relax_gamma_data_average[11]
    difference_FC5_gamma = power_push_gamma_data_average[3] - power_relax_gamma_data_average[3]
    difference_FC6_gamma= power_push_gamma_data_average[10] - power_relax_gamma_data_average[10]
    difference_T7_gamma = power_push_gamma_data_average[4] - power_relax_gamma_data_average[4]
    difference_T8_gamma = power_push_gamma_data_average[9] - power_relax_gamma_data_average[9]
    difference_P7_gamma = power_push_gamma_data_average[5] - power_relax_gamma_data_average[5]
    difference_P8_gamma = power_push_gamma_data_average[8] - power_relax_gamma_data_average[8]
    difference_O1_gamma = power_push_gamma_data_average[6] - power_relax_gamma_data_average[6]
    difference_O2_gamma= power_push_gamma_data_average[7] - power_relax_gamma_data_average[7]


#

##### average differences for channels and frequencies and times 
# i will divide all the time points in 3 parts, part I, part II, part III

# list of difference values we have 

    alpha_differences = [difference_AF3_alpha, difference_AF4_alpha, difference_F7_alpha, difference_F8_alpha, difference_F3_alpha, difference_F4_alpha, difference_FC5_alpha, difference_FC6_alpha, difference_T7_alpha, difference_T8_alpha, difference_P7_alpha, difference_P8_alpha, difference_O1_alpha, difference_O2_alpha]


# allpha differences part I 

    alpha1_differences_values = []

    for channel in alpha_differences:
        value = abs(channel[slice(0,199)].mean(axis=0))
        alpha1_differences_values.append(value)
    
# allpha differences part II
    
    alpha2_differences_values = []

    for channel in alpha_differences:
        value = abs(channel[slice(199,2*199)].mean(axis=0))
        alpha2_differences_values.append(value)
    
# allpha differences part III

    alpha3_differences_values = []

    for channel in alpha_differences:
        value = abs(channel[slice(2*199,3*199)].mean(axis=0))
        alpha3_differences_values.append(value)


    beta_differences = [difference_AF3_beta, difference_AF4_beta, difference_F7_beta, difference_F8_beta, difference_F3_beta, difference_F4_beta, difference_FC5_beta, difference_FC6_beta, difference_T7_beta, difference_T8_beta, difference_P7_beta, difference_P8_beta, difference_O1_beta, difference_O2_beta]


# allpha differences part I 

    beta1_differences_values = []

    for channel in beta_differences:
        value = abs(channel[slice(0,199)].mean(axis=0))
        beta1_differences_values.append(value)
    
# allpha differences part II

    beta2_differences_values = []

    for channel in beta_differences:
        value = abs(channel[slice(199,2*199)].mean(axis=0))
        beta2_differences_values.append(value)
    
# allpha differences part III

    beta3_differences_values = []

    for channel in beta_differences:
        value = abs(channel[slice(2*199,3*199)].mean(axis=0))
        beta3_differences_values.append(value)


    
    gamma_differences = [difference_AF3_gamma, difference_AF4_gamma, difference_F7_gamma, difference_F8_gamma, difference_F3_gamma, difference_F4_gamma, difference_FC5_gamma, difference_FC6_gamma, difference_T7_gamma, difference_T8_gamma, difference_P7_gamma, difference_P8_gamma, difference_O1_gamma, difference_O2_gamma]


# allpha differences part I 

    gamma1_differences_values = []

    for channel in gamma_differences:
        value = abs(channel[slice(0,199)].mean(axis=0))
        gamma1_differences_values.append(value)
    
# allpha differences part II

    gamma2_differences_values = []

    for channel in gamma_differences:
        value = abs(channel[slice(199,2*199)].mean(axis=0))
        gamma2_differences_values.append(value)
    
# allpha differences part III

    gamma3_differences_values = []

    for channel in gamma_differences:
        value = abs(channel[slice(2*199,3*199)].mean(axis=0))
        gamma3_differences_values.append(value)

    print("Alpha differences values in 1st chunk:\n", alpha1_differences_values)
    print("Alpha differences values in 2nd chunk:\n", alpha2_differences_values)
    print("Alpha differences values in 3rd chunk:\n", alpha3_differences_values)
    print("Beta differences values in 1st chunk:\n", beta1_differences_values)
    print("Beta differences values in 2nd chunk:\n", beta2_differences_values)
    print("Beta differences values in 3rd chunk:\n", beta3_differences_values)
    print("Gamma differences values in 1st chunk:\n", gamma1_differences_values)
    print("Gamma differences values in 2nd chunk:\n", gamma2_differences_values)
    print("Gamma differences values in 3rd chunk:\n", gamma3_differences_values)

    plt.figure(85)
    plt.plot(epochs_all.ch_names,alpha1_differences_values, label = "alpha")
    plt.plot(epochs_all.ch_names,beta1_differences_values, label = "beta")
    plt.plot(epochs_all.ch_names,gamma1_differences_values, label = "gamma")
    plt.xlabel("Channel")
    plt.ylabel("Power difference")
    plt.suptitle('Average differences in push/relax in 1 s - 3.34 s ', fontsize=16)
    plt.legend()
    plt.show

    plt.figure(86)
    plt.plot(epochs_all.ch_names,alpha2_differences_values, label = "alpha")
    plt.plot(epochs_all.ch_names,beta2_differences_values, label = "beta")
    plt.plot(epochs_all.ch_names,gamma2_differences_values, label = "gamma")
    plt.xlabel("Channel")
    plt.ylabel("Power difference")
    plt.suptitle('Average differences in push/relax in 3.34 s - 5.67 s ', fontsize=16)
    plt.legend()
    plt.show

    plt.figure(87)
    plt.plot(epochs_all.ch_names,alpha3_differences_values, label = "alpha")
    plt.plot(epochs_all.ch_names,beta3_differences_values, label = "beta")
    plt.plot(epochs_all.ch_names,gamma3_differences_values, label = "gamma")
    plt.xlabel("Channel")
    plt.ylabel("Power difference")
    plt.suptitle('Average differences in push/relax in 5.67 s - 8s ', fontsize=16)
    plt.legend()
    plt.show


# In[5]:


def create_TF_object_for_plotting(low_band, high_band, num, epochs1, epochs2):
    import numpy as np
    import mne
    freqs = np.logspace(*np.log10([low_band, high_band]), num=num)
    n_cycles = freqs / 2.

    power_data_push, itc_data_push = mne.time_frequency.tfr_morlet(epochs1, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                           return_itc=True, decim=3, n_jobs=1)
    power_data_relax, itc_data_relax = mne.time_frequency.tfr_morlet(epochs2, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                           return_itc=True, decim=3, n_jobs=1)
    
    
    return power_data_push, power_data_relax


# In[ ]:


def create_time_frequency_matrices(low_band, high_band, num, n_epochs, channels, n_samples, epochs):
    import numpy as np
    import mne
    # frequencies of interest
    freqs = np.logspace(*np.log10([low_band, high_band]), num=num)

    # empty matrices 
    power_data = np.zeros(shape=(n_epochs,channels, num, n_samples))

    itc_data = np.zeros(shape=(n_epochs,channels, num, n_samples))

    # number of epochs per class
    instances = np.arange(0,n_epochs,1)
    
    for i in instances:
        n_cycles = freqs / 2.

        power_data_single, itc_data_single = mne.time_frequency.tfr_morlet(epochs[i], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                           return_itc=True, decim=3, n_jobs=1)
    
        power_data[i,:,:,:] = power_data_single.data
        itc_data[i,:,:,:] = itc_data_single.data
    
    
    return  freqs, power_data, itc_data, instances


# In[10]:


def run_lda(class1, class2, n_epochs):
    
    
    import numpy as np
    # average over frequencies in the range 
    class1=class1.mean(axis=2)
    class2=class2.mean(axis=2)
    
    # flattening the input vectors
    
    class1 = class1.reshape(n_epochs, -1)
    class2 = class2.reshape(n_epochs, -1)
    
    # concatenating into one input vector
    
    input_data = np.concatenate((class1, class2), axis = 0)
    
    # creating labels
    
    push = np.ones(shape=(n_epochs))
    relax = np.zeros(shape=(n_epochs))
    labels = np.concatenate((push ,relax), axis = 0)
    
    
    # Now, we run the alpha data through LDA

    from sklearn.pipeline import Pipeline
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import ShuffleSplit, cross_val_score

    scores = []
    all_data = input_data
    all_data_train = input_data.copy()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(all_data_train)

# Assemble a classifier
    lda = LinearDiscriminantAnalysis()

# Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('LDA', lda)])
    scores = cross_val_score(clf, all_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("LDA Classification accuracy:", np.mean(scores))
    
    
    return input_data, scores


# In[8]:


def run_svm(class1, class2, n_epochs):
    import numpy as np

    # average over frequencies in the range 
    class1=class1.mean(axis=2)
    class2=class2.mean(axis=2)

    
    # concatenating into one input vector
    
    input_data = np.concatenate((class1, class2), axis = 0)
    
    # creating labels
    
    push = np.ones(shape=(n_epochs))
    relax = np.zeros(shape=(n_epochs))
    labels = np.concatenate((push ,relax), axis = 0)
    
    # flattening & normalizing the input data
    
    input_data = input_data.reshape(len(input_data), -1)
    input_data = input_data / np.std(input_data)
    
    
    # import the classifier and create a pipeline
    
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.model_selection import ShuffleSplit, cross_val_score

    # Define an SVM classifier (SVC) with a linear kernel
    clf = SVC(C=1, kernel='linear')

    
    # specifing cv 
    cv = ShuffleSplit(len(input_data), 10, test_size=0.2, random_state=42)

    # classifying the data
    scores_full = cross_val_score(clf, input_data, labels, cv=cv, n_jobs=1)
    
    # printing results
    print("SVM Classification score: %s (std. %s)" % (np.mean(scores_full), np.std(scores_full)))
    
    return input_data, scores_full


# In[3]:


def sliding_estimator(class1, class2, n_epochs, n_samples):
    
    import matplotlib.pyplot as plt
    # decoding over time - comparisons at every single time point for alpha frequency data 
    from sklearn.pipeline import make_pipeline
    from mne.decoding import Scaler, Vectorizer, cross_val_multiscore 
    from sklearn.preprocessing import StandardScaler
    from mne.decoding import SlidingEstimator
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import ShuffleSplit, cross_val_score
    # average over frequencies in the range 
    class1=class1.mean(axis=2)
    class2=class2.mean(axis=2)
    
    # creating input data
    
    input_data = np.concatenate((class1, class2), axis = 0)
    
    # creating labels
    
    push = np.ones(shape=(n_epochs))
    relax = np.zeros(shape=(n_epochs))
    labels = np.concatenate((push ,relax), axis = 0)

# creating X and y again

    X = input_data
    y = labels

# classifier pipeline; no need of vectorization here

    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())

# the sliding estimator will train the classifier at each time point

    scoring = "roc_auc"
    time_decoder = SlidingEstimator(clf, scoring=scoring, n_jobs=1, verbose=True)

# run cross-validation
# we want cross-validation without shuffling (block cross-validation)
# scikit learn does not shuffle by default 

    n_splits = 5
    scores = cross_val_multiscore(time_decoder, X, y, cv=5, n_jobs=1)

# mean and SD of ROC AUC across cross-validation runs
    mean_scores = np.mean(scores, axis=0)
    mean_across_all_times = round(np.mean(scores), 3)


    print (f"Mean cross-validation score across all timepoints: {mean_across_all_times:.3f}")


    fig, ax = plt.subplots()

    ax.axhline(0.5, color='k', linestyle='--', label='chance')  # AUC = 0.5
    ax.axvline(0, color='k', linestyle='-')  # Mark time point zero.
    ax.plot(np.arange(n_samples), mean_scores, label='score')

    ax.set_xlabel('Samples (SR of 85 Hz)')
    ax.set_ylabel('Mean ROC AUC')
    ax.legend()
    ax.set_title('Push vs Relax')
    fig.suptitle('Decoding')
    fig.show()
    
    return class1, class2, mean_scores


# In[ ]:


def crop_times(bottom_bound, upper_bound, class1, class2):


    class1 = class1[:,:,slice(bottom_bound,upper_bound)]
    class2 = class2[:,:,slice(bottom_bound,upper_bound)]
    return class1, class2


# In[3]:


def select_channels(channels, data, epochs, samples):
    
    import numpy as np
    if isinstance(channels, int) is True:
    
        length = 1
    
    else:
        
        length = len(channels)
    
    data_matrix = np.zeros(shape=(epochs, length, 20, samples))
    
    if isinstance(channels, int) is True:
    
        chosen_channel = data[:, channels, :, :]
        data_matrix[:, 0, :, :] = chosen_channel
    
    else:
        
        for i in channels:
    
            chosen_channel = data[:, i, :, :]
        
        
            data_matrix[:, channels.index(i), :, :] = chosen_channel 
    
    return data_matrix 


# In[5]:


def csp_lda(class1, class2, epochs):

    import numpy as np
    import matplotlib.pyplot as plt

    from mne.decoding import CSP
    from mne.time_frequency import AverageTFR

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder

    clf = make_pipeline(CSP(n_components=4, reg=None, log=True, norm_trace=False),
                    LinearDiscriminantAnalysis())
    n_splits = 5  # how many folds to use for cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    class1 = class1.mean(axis=2)
    class2 = class2.mean(axis=2)
    X = np.concatenate((class1,class2), axis=0)
    
    y = np.concatenate((np.ones(shape=(epochs)), np.zeros(shape=(epochs))), axis = 0)
    
    result = np.mean(cross_val_score(estimator=clf, X=X, y=y,
                                                scoring='roc_auc', cv=cv,
                                                n_jobs=1), axis=0)
    print(result)
    return result 


# In[11]:


# using pyEEG module to extract the features
# TO MODULARIZE THE CODE
def extract_pyeegfeatures(epochs_number, epochs_name1, epochs_name2):

    import pyeeg
    import numpy as np

    channels_list = np.arange(14)
    channels_list
    instances = np.arange(epochs_number)
    print(instances.shape)

    push_epochs_data = epochs_name1
    relax_epochs_data = epochs_name2

    push_epochs_data = push_epochs_data.get_data()
    relax_epochs_data = relax_epochs_data.get_data()


    #### Extracting DFA parameters

    dfa_per_channel_push = np.zeros(shape=(14))
    dfa_epochs_over_channels_push = np.zeros(shape=(epochs_number, 14))
    dfa_per_channel_relax = np.zeros(shape=(14))
    dfa_epochs_over_channels_relax = np.zeros(shape=(epochs_number, 14))

    for instance in instances:
    
        for channel in channels_list:
            dfa_push = pyeeg.dfa(push_epochs_data[instance, channel])
            dfa_per_channel_push[channel]=dfa_push
            dfa_relax = pyeeg.dfa(relax_epochs_data[instance, channel])
            dfa_per_channel_relax[channel]=dfa_relax
    
        dfa_epochs_over_channels_push[instance] = dfa_per_channel_push
        dfa_epochs_over_channels_relax[instance] = dfa_per_channel_relax

    print(dfa_epochs_over_channels_push.shape)
    print(dfa_epochs_over_channels_relax.shape)

    #### Extracting Fisher Information

    fi_per_channel_push = np.zeros(shape=(14))
    fi_epochs_over_channels_push = np.zeros(shape=(epochs_number, 14))
    fi_per_channel_relax = np.zeros(shape=(14))
    fi_epochs_over_channels_relax = np.zeros(shape=(epochs_number, 14))

    for instance in instances:
    
        for channel in channels_list:
            fi_push = pyeeg.fisher_info(push_epochs_data[instance, channel],1, 2)
            fi_per_channel_push[channel]=fi_push
            fi_relax = pyeeg.fisher_info(relax_epochs_data[instance, channel], 1,2)
            fi_per_channel_relax[channel]=fi_relax
    
        fi_epochs_over_channels_push[instance] =  fi_per_channel_push
        fi_epochs_over_channels_relax[instance] =  fi_per_channel_relax

    print(fi_epochs_over_channels_push.shape)
    print(fi_epochs_over_channels_relax.shape)
    
    #### Extracting Hurst Exponent

    pyeeg.hurst(push_epochs_data[6, 4])

    he_per_channel_push = np.zeros(shape=(14))
    he_epochs_over_channels_push = np.zeros(shape=(epochs_number, 14))
    he_per_channel_relax = np.zeros(shape=(14))
    he_epochs_over_channels_relax = np.zeros(shape=(epochs_number, 14))

    for instance in instances:
    
        for channel in channels_list:
            he_push = pyeeg.hurst(push_epochs_data[instance, channel])
            he_per_channel_push[channel]=he_push
            he_relax = pyeeg.hurst(push_epochs_data[instance, channel])
            he_per_channel_relax[channel]=he_relax
    
        he_epochs_over_channels_push[instance] =  he_per_channel_push
        he_epochs_over_channels_relax[instance] =  he_per_channel_relax

    print(he_epochs_over_channels_push.shape)
    print(he_epochs_over_channels_relax.shape)

    #### Extracting Hjorth Fractal Dimension 

    ### according to a paper saved in my feature extraction folder, value of 18 is best for kmax argument of hfd

    pyeeg.hfd(push_epochs_data[6, 4],18)

    hfd_per_channel_push = np.zeros(shape=(14))
    hfd_epochs_over_channels_push = np.zeros(shape=(epochs_number, 14))
    hfd_per_channel_relax = np.zeros(shape=(14))
    hfd_epochs_over_channels_relax = np.zeros(shape=(epochs_number, 14))

    for instance in instances:
    
        for channel in channels_list:
            hfd_push = pyeeg.hfd(push_epochs_data[instance, channel], 18)
            hfd_per_channel_push[channel]=hfd_push
            hfd_relax = pyeeg.hfd(push_epochs_data[instance, channel], 18)
            hfd_per_channel_relax[channel]=hfd_relax
    
        hfd_epochs_over_channels_push[instance] =  hfd_per_channel_push
        hfd_epochs_over_channels_relax[instance] =  hfd_per_channel_relax

    print(hfd_epochs_over_channels_push.shape)
    print(hfd_epochs_over_channels_relax.shape)

    #### Extracting Hjorth Mobility & Complexity


    # first is mobility
    # second is complexity

    hm_per_channel_push = np.zeros(shape=(14))
    hm_epochs_over_channels_push = np.zeros(shape=(epochs_number, 14))
    hc_per_channel_push = np.zeros(shape=(14))
    hc_epochs_over_channels_push = np.zeros(shape=(epochs_number, 14))
    hm_per_channel_relax = np.zeros(shape=(14))
    hm_epochs_over_channels_relax = np.zeros(shape=(epochs_number, 14))
    hc_per_channel_relax = np.zeros(shape=(14))
    hc_epochs_over_channels_relax = np.zeros(shape=(epochs_number, 14))

    for instance in instances:
    
        for channel in channels_list:
            hmc_push = pyeeg.hjorth(push_epochs_data[instance, channel])
            hm_per_channel_push[channel]=hmc_push[0]
            hc_per_channel_push[channel]=hmc_push[1]
            hmc_relax = pyeeg.hjorth(push_epochs_data[instance, channel])
            hm_per_channel_relax[channel]=hmc_relax[0]
            hc_per_channel_relax[channel]=hmc_relax[1]
    
        hm_epochs_over_channels_push[instance] =  hm_per_channel_push
        hc_epochs_over_channels_push[instance] =  hc_per_channel_push
        hm_epochs_over_channels_relax[instance] =  hm_per_channel_relax
        hc_epochs_over_channels_relax[instance] =  hc_per_channel_relax

    print(hm_epochs_over_channels_push.shape)
    print(hc_epochs_over_channels_relax.shape)
    print(hm_epochs_over_channels_push.shape)
    print(hc_epochs_over_channels_relax.shape)

    #### Extracting Petrosian Fractal Dimension


    pfd_per_channel_push = np.zeros(shape=(14))
    pfd_epochs_over_channels_push = np.zeros(shape=(epochs_number, 14))
    pfd_per_channel_relax = np.zeros(shape=(14))
    pfd_epochs_over_channels_relax = np.zeros(shape=(epochs_number, 14))

    for instance in instances:
    
        for channel in channels_list:
            pfd_push = pyeeg.pfd(push_epochs_data[instance, channel])
            pfd_per_channel_push[channel]=pfd_push
            pfd_relax = pyeeg.pfd(push_epochs_data[instance, channel])
            pfd_per_channel_relax[channel]=pfd_relax
    
        pfd_epochs_over_channels_push[instance] =  pfd_per_channel_push
        pfd_epochs_over_channels_relax[instance] =  pfd_per_channel_relax

    print(pfd_epochs_over_channels_push.shape)
    print(pfd_epochs_over_channels_relax.shape)
    
    push_pyeeg_features = np.concatenate((fi_epochs_over_channels_push, dfa_epochs_over_channels_push, he_epochs_over_channels_push, hfd_epochs_over_channels_push, hm_epochs_over_channels_push, hc_epochs_over_channels_push, pfd_epochs_over_channels_push), axis = 1)
    push_pyeeg_features.shape

    relax_pyeeg_features = np.concatenate((fi_epochs_over_channels_push, dfa_epochs_over_channels_push, he_epochs_over_channels_push, hfd_epochs_over_channels_push, hm_epochs_over_channels_push, hc_epochs_over_channels_push, pfd_epochs_over_channels_push), axis = 1)
    relax_pyeeg_features.shape

    all_pyeeg_features =  np.concatenate((push_pyeeg_features, relax_pyeeg_features), axis = 0)
    all_pyeeg_features.shape

    

    return fi_epochs_over_channels_push, dfa_epochs_over_channels_push, he_epochs_over_channels_push, hfd_epochs_over_channels_push, hm_epochs_over_channels_push, hc_epochs_over_channels_push, pfd_epochs_over_channels_push, fi_epochs_over_channels_push, dfa_epochs_over_channels_push, he_epochs_over_channels_push, hfd_epochs_over_channels_push, hm_epochs_over_channels_push, hc_epochs_over_channels_push, pfd_epochs_over_channels_push, fi_epochs_over_channels_relax, dfa_epochs_over_channels_relax, he_epochs_over_channels_relax, hfd_epochs_over_channels_relax, hm_epochs_over_channels_relax, hc_epochs_over_channels_relax, pfd_epochs_over_channels_relax,all_pyeeg_features 


# In[ ]:




