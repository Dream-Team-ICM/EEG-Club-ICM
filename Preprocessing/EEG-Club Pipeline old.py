#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 16:39:24 2023

@author: marie.degrave
"""


# importing packages
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import (ICA, corrmap, create_ecg_epochs,
                               create_eog_epochs)

# In[2]:


# Example 1: Raw EGI files

data_raw_file = os.path.join(
    '/Users/marie.degrave/Documents/GitHub/EEG-Club-ICM/N170 All Data and Scripts/1/1_N170.set/'
    )
raw = mne.io.read_raw_eeglab(data_raw_file, preload=True)

print(raw)



# In[11]:   Exploring my data


# Exploring metadata
n_time_samps = raw.n_times
time_secs = raw.times
ch_names = raw.ch_names
n_chan = len(ch_names)  # note: there is no raw.n_channels attribute
print('the data object has {} time samples and {} channels.'
      ''.format(n_time_samps, n_chan))
print('The last time sample is at {} seconds.'.format(time_secs[-1]))
print('The first few channel names are {}.'.format(', '.join(ch_names[:3])))
print()  # insert a blank line in the output

# some examples of raw.info:
print('bad channels:', raw.info['bads'])  # chs marked "bad" during acquisition
print(raw.info['sfreq'], 'Hz')            # sampling frequency
print(raw.info['description'], '\n')      # miscellaneous acquisition info

print(raw.info)

# annotations structure 
set(raw.annotations.description)
set(raw.annotations.duration)
raw.annotations.onset


# In[13]:   Select channels


#stim sig: 
# {
#   "value": {
#     "LongName": "Event code value",
#     "Levels": {
# 	   "1-40": "Stimulus - faces",
# 	   "41-80": "Stimulus - cars",
# 	   "101-140": "Stimulus - scrambled faces",
# 	   "141-180": "Stimulus - scrambled cars",
# 	   
# 	   "201": "Response - correct",
# 	   "202": "Response - error"
#     }
#   }
# }

# selecting data (channels and time points)  ---> EOG ici
eeg_and_eog = raw.copy().pick_types(eeg=True, eog=True, emg=False)
print('Selecting EEG and EOG channels only:')
print(len(raw.ch_names), '→', len(eeg_and_eog.ch_names))


# stim_channels = raw.copy().pick_types(stim=True)
# print('Selecting Stim channels only:')
# print(len(raw.ch_names), '→', len(stim_channels.ch_names))


raw_selection = raw.copy().crop(tmin=0, tmax=60)
print(raw_selection)


# In[25]: !!!!!!!!!!!


# exploring events
print(stim_channels.ch_names)
events_stim = mne.find_events(raw,stim_channel=['StOn'])


# In[27]:   Plot data


# get and plot data
raw.plot(events=events_stim, start=10, duration=20, color='gray')


# In[40]:   Notch filter


# line noise
eeg_only = raw.copy().pick_types(eeg=True, eog=False, emg=False)
fig = eeg_only.plot_psd(fmax=100, average=True)

#psd_welch = mne.time_frequency.psd_welch(eeg_only,fmax=100) 
#plt.plot(psd_welch[1],np.log(np.mean(psd_welch[0],0)))


# In[51]:  Filtering


# quick filtering
freqs = (50, 100)
eeg_only.load_data()
eeg_only_notch = eeg_only.copy().notch_filter(freqs=freqs)
fig_notch = eeg_only_notch.plot_psd(fmax=100, average=True)

fig_time =eeg_only_notch.plot(scalings=dict(eeg=200e-6), duration=20*60)

# more filtering
eeg_only_highpass_notch = eeg_only_notch.copy().filter(l_freq=0.1, h_freq=30)
fig_time =eeg_only_highpass_notch.plot(scalings=dict(eeg=150e-6), duration=20*60)


# In[]:  Re-reference data ?

raw = raw.set_eeg_reference(['Oz'])


# In[]:  Inspect electrodes and reject noisy channels




# In[]:  Create EOG channels to do the ICA afterward

#create HEOG channel
heog_info = mne.create_info(['HEOG'], 256, "eog")
heog_data = raw['HEOG_left'][0]-raw['HEOG_right'][0]
heog_raw = mne.io.RawArray(heog_data, heog_info)
# create VEOG channel
veog_info = mne.create_info(['VEOG'], 256, "eog")
veog_data = raw['VEOG_lower'][0]-raw['FP2'][0]
veog_raw = mne.io.RawArray(heog_data, veog_info)
#Append them to the data
raw.add_channels([heog_raw, veog_raw],True)



# In[]:  Create a montage to maps channels to a layout

from mne.io import read_raw_eeglab, read_raw
from mne.channels import read_dig_polhemus_isotrak, read_custom_montage

# 1 - load montage with all possible possitions
montage = read_custom_montage('standard_10_5_cap385.elp')
# 2 - correct FP names
raw.rename_channels(dict(FP1 = 'Fp1', FP2 = 'Fp2'))

# 3 - make dict of channel positions
ch_positions = dict()
for ch in raw.ch_names:
     if not (ch in ['VEOG_lower', 'HEOG_right', 'HEOG_left']):
        ch_index = montage.ch_names.index(ch)+3
        ch_positions.update({ch : montage.dig[ch_index]['r']})

# 4 - create montage with really occuring channels in our data
montage = mne.channels.make_dig_montage(ch_positions,
                             nasion = montage.dig[1]['r'],
                             lpa = montage.dig[0]['r'],
                             rpa = montage.dig[2]['r'])

# 5 add it to the raw object
raw.set_montage(montage, on_missing='ignore')


# In[]:  Run ICA and reject noisy components


#Ex EOG ---> Bizarre
raw.crop(0, 60).load_data()
eog_evoked = create_eog_epochs(raw).average() 
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()











