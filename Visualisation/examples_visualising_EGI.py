#!/usr/bin/env python
# coding: utf-8

# In[6]:


# importing packages
import os
import numpy as np
import matplotlib.pyplot as plt
import mne


# In[2]:


# Example 1: Raw EGI files
egi_data_raw_file = os.path.join('/Users/marie.degrave/Documents/EEG-Club/Session 3 Visualize data/CEM005JS.raw')
raw=mne.io.read_raw_egi(egi_data_raw_file)
print(raw)


# In[11]:


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


# In[13]:


# selecting data (channels and time points)
eeg_and_eog = raw.copy().pick_types(eeg=True, eog=True, emg=False)
print('Selecting EEG and EOG channels only:')
print(len(raw.ch_names), '→', len(eeg_and_eog.ch_names))

stim_channels = raw.copy().pick_types(stim=True)
print('Selecting Stim channels only:')
print(len(raw.ch_names), '→', len(stim_channels.ch_names))

raw_selection = raw.copy().crop(tmin=0, tmax=60)
print(raw_selection)


# In[25]:


# exploring events
print(stim_channels.ch_names)
events_stim = mne.find_events(raw,stim_channel=['StOn'])


# In[27]:


# get and plot data
raw.plot(events=events_stim, start=10, duration=20, color='gray')


# In[40]:


# line noise
eeg_only = raw.copy().pick_types(eeg=True, eog=False, emg=False)
fig = eeg_only.plot_psd(fmax=100, average=True)

#psd_welch = mne.time_frequency.psd_welch(eeg_only,fmax=100) 
#plt.plot(psd_welch[1],np.log(np.mean(psd_welch[0],0)))


# In[51]:


# quick filtering
freqs = (50, 100)
eeg_only.load_data()
eeg_only_notch = eeg_only.copy().notch_filter(freqs=freqs)
fig_notch = eeg_only_notch.plot_psd(fmax=100, average=True)

fig_time =eeg_only_notch.plot(scalings=dict(eeg=200e-6), duration=20*60)


# In[53]:


# more filtering
eeg_only_highpass_notch = eeg_only_notch.copy().filter(l_freq=0.1, h_freq=30)
fig_time =eeg_only_highpass_notch.plot(scalings=dict(eeg=150e-6), duration=20*60)

