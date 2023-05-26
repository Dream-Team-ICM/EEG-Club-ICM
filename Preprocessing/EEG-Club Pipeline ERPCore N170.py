# -*- coding: utf-8 -*-
"""
===============================================================================
Spyder Editor
author = Marie Degrave & Thomas Andrillon (2023)

ICM EEG Club Example pipeline
Data: ERP Core MMN paradigm

===============================================================================
"""

# %% IMPORT MODULES
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import glob
from autoreject import AutoReject
from mne_icalabel import label_components

#from copy import deepcopy

# %% Paths & Variables
# Paths
if 'thandrillon' in os.getcwd():
    path_data='/Users/thandrillon/Data/ERPCore/ERPCore_N170/'
elif 'degrave' in os.getcwd():
    path_data='your_path'
else:
    path_data='your_path'

if os.path.exists(path_data+"reports")==False:
    os.makedirs(path_data+"reports")
if os.path.exists(path_data+"intermediary")==False:
    os.makedirs(path_data+"intermediary")
    
files = glob.glob(path_data + '*.set')

# %% LOAD, FILTER, CLEAN
report_Event = mne.Report(title='Auto Reject')
report_AR = mne.Report(title='Auto Reject')
report_ERP = mne.Report(title='ERP')
report_ICA = mne.Report(title='ICA')

for file in files:

    # [1] LOAD RAW DATA
    file_name=file.split('/')[-1]
    sub_ID=file_name.split('_')[0]
    report_prefix=path_data+"reports/"+sub_ID+"_"
    raw = mne.io.read_raw_eeglab(file, preload=True)
    raw_eeg = raw.copy().drop_channels(['HEOG_left','HEOG_right','VEOG_lower'])
    
    
    print(raw_eeg)
    #HCGSN256_montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
    #raw.set_montage(HCGSN256_montage)

    # [2] MONTAGE
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_eeg.rename_channels(dict(FP1 = 'Fp1', FP2 = 'Fp2'))
    raw_eeg.set_montage(montage, on_missing='ignore')

    
    # [3] REREFERNCING AND FILTERING
    raw_eeg.resample(256)
    sfreq = raw_eeg.info["sfreq"]
    raw_eeg.set_eeg_reference("average")
    raw_eeg.filter(0.1, 100, fir_design='firwin')
    raw_eeg.notch_filter(60,
                    filter_length='auto', phase='zero')
    report = mne.Report(title=sub_ID)
    report.add_raw(raw=raw_eeg, title='Filtered Cont from"raw_eeg"', psd=False)  # omit PSD plot

    # [4] EVENTS
    stim_events=[range(1,40),range(41,80),range(101,140),range(141,180)]
    # face, car, scrambled face, scrambled car
    (events, event_dict) = mne.events_from_annotations(raw_eeg)
    events=mne.merge_events(events, list(range(1,41)), 1001, replace_events=True)
    events=mne.merge_events(events, list(range(41,81)), 1002, replace_events=True)
    events=mne.merge_events(events, list(range(101,141)), 1003, replace_events=True)
    events=mne.merge_events(events, list(range(141,181)), 1004, replace_events=True)
    
    count_events=mne.count_events(events)
    face_id=1001;
    car_id=1002;
    stim_events = mne.pick_events(events, include=[face_id,car_id])

    report.add_events(events=stim_events, title='Events from "stim_events"', sfreq=sfreq)

    report_Event.add_events(events=stim_events, title='Events: '+sub_ID, sfreq=sfreq)

    # [5] EPOCHS
    epochs = mne.Epochs(raw_eeg, events, event_id=[face_id,car_id],
                        tmin=-0.2, tmax=1.2, reject=None, preload=True)
    report.add_epochs(epochs=epochs, title='Epochs from "epochs"')
    savename = "e_" + sub_ID + ".fif"
    epochs.save(path_data+"intermediary/"+savename, overwrite=True)
    
    # [6] AUTOREJECT
    n_interpolates = np.array([1, 4, 32])
    consensus_percs = np.linspace(0, 1.0, 11)
    ar = AutoReject(n_interpolates, consensus_percs,
                    thresh_method='random_search', random_state=42)
    ar.fit(epochs)
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)
    epochs_clean.set_eeg_reference("average")

    savename = "ce_" + sub_ID + ".fif"
    epochs_clean.save(path_data+"intermediary/"+savename, overwrite=True)
    
    fig = reject_log.plot(orientation = 'horizontal', show=False)

    report.add_figure(
        fig=fig,
        title="Reject log",
        caption="The rejct log returned by autoreject",
        image_format="PNG",
    )
    report_AR.add_figure(
        fig=fig,
        title="Reject log: "+sub_ID,
        caption="The rejct log returned by autoreject",
        image_format="PNG",
    )
    
    # [8] ICA
    ica_epochs = mne.Epochs(raw_eeg.copy().filter(l_freq=1.0, h_freq=None), events, event_id=[face_id,car_id],
                        tmin=-0.2, tmax=1.2, reject=None, preload=True,baseline=None)
    ica_epochs_clean = ar.transform(ica_epochs)
    ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=97,method='infomax', fit_params=dict(extended=True))
    ica.fit(ica_epochs_clean)
    savename = "ica_ce_" + sub_ID + ".fif"
    ica.save(path_data+"intermediary/"+savename, overwrite=True)
    
    # ICA rejection
    ica_classification=label_components(ica_epochs_clean, ica, method='iclabel')
    ica_labels=pd.DataFrame(ica_classification)
    ica_labels.to_csv(report_prefix+'ICAlabels.csv')
    labels = ica_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    caption_str=''
    for idx, label in enumerate(labels):
        caption_str=caption_str+'ICA'+str(idx)+': '+label+'; '
        
    print(f"Excluding these ICA components: {exclude_idx}")
    epochs_clean_ica=ica.apply(epochs_clean, exclude=exclude_idx)
    epochs_clean_ica.set_eeg_reference("average")

    ica_fig=ica.plot_sources(ica_epochs_clean, show_scrollbars=True, show=False)
    report.add_ica(
        ica=ica,
        title='ICA cleaning',
        inst=ica_epochs_clean,
        n_jobs=2  # could be increased!
    )
    report.add_figure(
        ica_fig,
        title="ICA sources",
        caption=caption_str,
    )
    
    report_ICA.add_ica(
        ica=ica,
        title='ICA:' + sub_ID,
        inst=ica_epochs_clean,
        n_jobs=2  # could be increased!
    )
    report_ICA.add_figure(
        ica_fig,
        title='ICA sources:' + sub_ID,
        caption=caption_str,
    )

    # [7] ERP
    evoked_face  = epochs[str(face_id)].average()
    evoked_car = epochs[str(car_id)].average()
    
    evoked_clean_face  = epochs_clean[str(face_id)].average()
    evoked_clean_car = epochs_clean[str(car_id)].average()
    
    evoked_ica_clean_face  = epochs_clean_ica[str(face_id)].average()
    evoked_ica_clean_car = epochs_clean_ica[str(car_id)].average()
    
    conditions=[str(face_id),str(car_id)];
    evoked_clean_perCond = {c:epochs_clean_ica[c].average() for c in conditions}
    savename = "erp_ce_" + sub_ID + ".fif"
    mne.write_evokeds(path_data+"intermediary/"+savename, 
                      list(evoked_clean_perCond.values()), overwrite=True
                     )
    report.add_evokeds(
        evokeds=[evoked_car,evoked_face,evoked_clean_car,evoked_clean_face,evoked_ica_clean_car,evoked_ica_clean_face],
        titles=["car", "face","clean car", "clean face","ica+clean car", "ica+clean face"],  # Manually specify titles
        n_time_points=5,
        replace=True)

    
    # [9] CONTRAST
    picks = 'PO8'
    evokeds_ica_clean = dict(car=evoked_ica_clean_face, face=evoked_ica_clean_car)
    erp_ica_clean_fig=mne.viz.plot_compare_evokeds(evokeds_ica_clean, picks=picks, show=False)
    evokeds = dict(car=evoked_face, face=evoked_car)
    erp_fig=mne.viz.plot_compare_evokeds(evokeds, picks=picks, show=False)

    report.add_figure(
         erp_fig,
         title="ERP contrast",
         caption="Face vs Car at PO8",
     )
    report.add_figure(
          erp_ica_clean_fig,
          title="ERP contrast cleaned+ica",
          caption="Face vs Car at PO8",
      )
 
    face_ica_clean_vis = mne.combine_evoked([evoked_ica_clean_face, evoked_ica_clean_car], weights=[1, -1])
    erp_ica_clean_but_fig=face_ica_clean_vis.plot_joint(show=False)
    face_clean_vis = mne.combine_evoked([evoked_clean_face, evoked_clean_car], weights=[1, -1])
    erp_clean_but_fig=face_clean_vis.plot_joint(show=False)
    face_vis = mne.combine_evoked([evoked_face, evoked_car], weights=[1, -1])
    erp_but_fig=face_vis.plot_joint(show=False)
    report.add_figure(
          erp_but_fig,
          title="ERP contrast (butterfly)",
          caption="Face vs Car across all Elec",
      )
    report.add_figure(
          erp_clean_but_fig,
          title="clean ERP contrast (butterfly)",
          caption="Face vs Car across all Elec",
      )
    report.add_figure(
          erp_ica_clean_but_fig,
          title="ica+clean ERP contrast (butterfly)",
          caption="Face vs Car across all Elec",
      )
    savename = "cont_ce_" + sub_ID + ".fif"
    face_ica_clean_vis.save(path_data+"intermediary/"+savename, overwrite=True)
    

    report_ERP.add_figure(
          erp_but_fig,
          title="diff: "+sub_ID,
          caption="Face vs Car across all Elec",
      )
    report_ERP.add_figure(
          erp_clean_but_fig,
          title="cleaned diff: "+sub_ID,
          caption="Face vs Car across all Elec",
      )
    report_ERP.add_figure(
          erp_ica_clean_but_fig,
          title="ica+cleaned diff: "+sub_ID,
          caption="Face vs Car across all Elec",
      )
    
    report.save(report_prefix+"pipeline.html", overwrite=True, open_browser=False)
    
    report_Event.save(path_data+"reports/"+"Events.html", overwrite=True, open_browser=False)
    report_AR.save(path_data+"reports/"+"AutoRej.html", overwrite=True, open_browser=False)
    report_ERP.save(path_data+"reports/"+"ERP.html", overwrite=True, open_browser=False)
    report_ICA.save(path_data+"reports/"+"ICA.html", overwrite=True, open_browser=False)
    
    plt.close('all')
    
# %% GET ERPs across subjects
evokeds_files = glob.glob(path_data+"intermediary/" + '/erp_ce_*.fif')
evokeds = {} #create an empty dict
conditions = ['1001','1002']
# #convert list of evoked in a dict (w/ diff conditions if needed)
for idx, c in enumerate(conditions):
    evokeds[c] = [mne.read_evokeds(d)[idx] for d in 
    evokeds_files]

evokeds # We can see that he matched the conditions by treating each as if it was 2 objcts as before 


# "Plot averaged ERP on all subj"
ERP_mean = mne.viz.plot_compare_evokeds(evokeds,
                             picks='PO8', show_sensors='upper right',
                             title='Averaged ERP all subjects',
                            )
plt.show()


#gfp: "Plot averaged ERP on all subj"
ERP_gfp = mne.viz.plot_compare_evokeds(evokeds,
                             combine='gfp', show_sensors='upper right',
                             title='Averaged ERP all subjects',
                            )
plt.show()


# evokeds_files = glob.glob(path_data+"intermediary/" + '/cont_ce_*.fif')
# evokeds_diff = {} #create an empty dict
# # #convert list of evoked in a dict (w/ diff conditions if needed)
# for idx, d in enumerate(evokeds_files):
#     evokeds_diff[idx] = mne.read_evokeds(d)[0]
    
# ERP_mean = mne.viz.plot_evoked(evokeds_diff,
#                              picks='PO8',
#                              title='Averaged difference wave all subjects',
#                             )
# plt.show()