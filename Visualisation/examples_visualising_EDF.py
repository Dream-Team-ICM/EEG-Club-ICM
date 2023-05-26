#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing packages
import os
import numpy as np
import matplotlib.pyplot as plt
import mne


# In[2]:


# Example 1: Raw EGI files
egi_data_raw_file = os.path.join('/Users/thandrillon/Downloads',
                                    'Ex_EDF.edf')
raw=mne.io.read_raw_edf(egi_data_raw_file)
print(raw)


# In[3]:


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
print(raw.get_channel_types())

# In[4]:


# selecting data 
channel_names = ['Fp1-A2', 'C3-A2', 'O1-A2']
eeg_and_eog = raw.copy().pick(channel_names)

# quick filtering
freqs = (50, 100)
eeg_and_eog.load_data()
eeg_and_eog_notch = eeg_and_eog.copy().notch_filter(freqs=freqs)
eeg_and_eog_bp = eeg_and_eog_notch.copy().filter(l_freq=0.1, h_freq=30)

fig_time =eeg_and_eog_bp.plot(scalings=dict(eeg=150e-6), duration=20, start=5230)

eeg_and_eog_bp2 = eeg_and_eog_notch.copy().filter(l_freq=1, h_freq=30)
fig_time2 =eeg_and_eog_bp2.plot(scalings=dict(eeg=150e-6), duration=20, start=5230)


eeg_and_eog_bp3 = eeg_and_eog_notch.copy().filter(l_freq=0.1, h_freq=15)
fig_time3 =eeg_and_eog_bp3.plot(scalings=dict(eeg=150e-6), duration=20, start=5230)

eeg_and_eog_bp3.set_eeg_reference(ref_channels=['Fp1-A2'])
fig_time3 =eeg_and_eog_bp3.plot(scalings=dict(eeg=150e-6), duration=20, start=5230)

# In[5]:
    
egi_data_raw_file2 = os.path.join('/Users/thandrillon/Downloads',
                                       'Ex_EDF_Clipped.edf')
raw2=mne.io.read_raw_edf(egi_data_raw_file2)
channel_names = ['Fp1-A2', 'C3-A2', 'O1-A2', 'EOG D','EOG G']
eeg_and_eog2 = raw2.copy().pick(channel_names)

# quick filtering
freqs = (50, 100)
eeg_and_eog2.load_data()
eeg_and_eog_notch2 = eeg_and_eog2.copy().notch_filter(freqs=freqs)
eeg_and_eog_bp2 = eeg_and_eog_notch2.copy().filter(l_freq=1, h_freq=30)

fig_time =eeg_and_eog_bp2.plot(scalings=dict(eeg=150e-6), duration=20, start=3435)
