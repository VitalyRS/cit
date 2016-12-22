
# coding: utf-8

# In[13]:

import pandas as pd
import os 
import time
import sys
import re
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
#https://www.kaggle.com/c/grasp-and-lift-eeg-detection
#the *_data.csv files contain the raw 32 channels EEG data (sampling rate 500Hz)
from IPython.display import clear_output
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
import mne


# In[3]:

folder="/home/vitaly/anaconda2/vit/DATA/GAL/train"
f1="subj1_series1_data.csv"
f2="subj1_series1_events.csv"
f1=folder+"/"+f1
f2=folder+"/"+f2


# In[ ]:

allfiles=" ".join(os.listdir(folder))
import re
pattern=u"\S+events\S+"
files_events=re.findall(pattern,allfiles)
pattern=u"\S+data\S+"
files_data=re.findall(pattern,allfiles)
print len(files_data)
files_events
pattern=u"\d*_series\d*"
files_data=re.findall(pattern,allfiles)
files_data
print len(set(files_data))
tt=list(set(files_data))
#'subj7_series1_events.csv', 'subj8_series3_data.csv'),
f1=map(lambda x:"subj"+x+"_events.csv",tt)
f2=map(lambda x:"subj"+x+"_data.csv",tt)
data={"file_events":f1,"file_data":f2}
pd.DataFrame(data)


# In[42]:


class DataManipulation(object):
    
    __sfreq=500 # 500 Hz frequncy sampling
    __ch_names=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4',
 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
      #channel names from data descr
    
    def __init__(self,folder_name):
            self.folder_name = folder_name
            allfiles=" ".join(os.listdir(self.folder_name))
            pattern=u"\d*_series\d*"
            files_data=re.findall(pattern,allfiles)
            tt=list(set(files_data))
            #'subj7_series1_events.csv', 'subj8_series3_data.csv'),
            f1=map(lambda x:"subj"+x+"_events.csv",tt)
            f2=map(lambda x:"subj"+x+"_data.csv",tt)
            data={"file_events":f1,"file_data":f2}
            bd=pd.DataFrame(data)
            self.bd=bd
            self.N=bd.shape[0]
            
    def print_files(self):
        return zip(self.files_events,self.files_data)
    def clean_data_checking(self):
        file_data_ok=[]
        file_events_ok=[]
        i=0
        print "checking data csv files..."
        for file_data in self.bd.file_data:
            
            try:
                pd.read_csv(self.folder_name+"/"+file_data)
                file_data_ok.append(1)
            except:
                print "exp"
                file_data_ok.append(0)
            time.sleep(0.01)
            #clear_output()
            update_progress(1.0*i/(self.N-1))
            i=i+1

            
        i=0    
        print "checking events csv files..."
        for file_events in self.bd.file_events:
            try:
                file_events_ok.append(1)
                pd.read_csv(self.folder_name+"/"+file_events)
            except:
                print "exp"
                file_events_ok.append(0)
            time.sleep(0.01)
            #clear_output()
            update_progress(1.0*i/(self.N-1))
            i=i+1

        self.bd["data_ok"]=file_data_ok
        self.bd["events_ok"]=file_events_ok
    def read_file(self,numero=0,log=0):
        if log==0: # reading data file
            return pd.read_csv(self.folder_name+"/"+self.bd.file_data[numero])
        else: # reading events file
            return pd.read_csv(self.folder_name+"/"+self.bd.file_events[numero])
    def read_and_norm(self,numero=0,log=0,fres=50,fr1=1,fr2=24):
        #numero
        #log
        #fres
        #fr1
        #fr2
        data=self.read_file(numero,log)
        data=data.values[:,1:]
        info = mne.create_info(  ch_names=self.__ch_names,       sfreq=self.__sfreq    )
        data = mne.io.RawArray(data.T, info,verbose=False)
        
        for names in data.ch_names:
            data.set_channel_types({names:'eeg'})
        
        montage = mne.channels.read_montage('standard_1020')
        data.set_montage(montage,verbose=None)
        clear_output()
        
        
        data, _ = mne.io.set_eeg_reference(data) # again average rereference
        data.resample(fres, npad="auto",verbose=None)  # set sampling frequency to 145Hz
        data.filter(fr1,fr2,h_trans_bandwidth='auto', filter_length='auto',
               phase='zero',verbose=None)
        return data
        

d=DataManipulation(folder)
print d.N



# In[45]:

data=d.read_and_norm()


# In[5]:

bd=d.read_file(0,0)
bd.head()


# In[9]:

list(bd.columns[1:])
ch_names=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4',
 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
data=bd.values[:,1:]


# In[10]:

print "16 channels"
sfreq=500
info = mne.create_info(
ch_names=ch_names,
sfreq=sfreq
    )

custom_raw = mne.io.RawArray(data.T, info,verbose=False)

data=custom_raw

for names in data.ch_names:
    data.set_channel_types({names:'eeg'})
        
montage = mne.channels.read_montage('standard_1020')
data.set_montage(montage,verbose=None)
clear_output()


# In[46]:

data.plot(scalings="auto")#, duration=80.0)
plt.show()


# In[49]:

data.plot_projs_topomap()


# In[ ]:

def create_mne_data(data,sfreq):
    print "16 channels"
    info = mne.create_info(
    ch_names=['Fp1','Fpz', 'Fp2', 'F7', 'F3', 'Fz',
             'F4', 'F8', 'T3', 'C3', 'Cz',
             'C4', 'T4', 'P3', 'Pz', 'P4'],
    sfreq=sfreq
    )

    custom_raw = mne.io.RawArray(data.T, info,verbose=False)

    data=custom_raw

    for names in data.ch_names:
        data.set_channel_types({names:'eeg'})
        
    montage = mne.channels.read_montage('standard_1020')
    data.set_montage(montage,verbose=None)
    return data
def data_norm(data,fres,fr1,fr2):
    data, _ = mne.io.set_eeg_reference(data) # again average rereference
    data.resample(fres, npad="auto",verbose=None)  # set sampling frequency to 145Hz
    data.filter(fr1,fr2,h_trans_bandwidth='auto', filter_length='auto',
           phase='zero',verbose=None)
#    data.notch_filter(50, filter_length='auto',
#                 phase='zero')
    return data


# In[ ]:

map(lambda x:pd.read_csv(d.folder_name+"/"+x),bd.file_data)


# In[ ]:

map(lambda x:pd.read_csv(d.folder_name+"/"+x),bd.file_events)


# In[ ]:

bdf1=pd.read_csv(f1)
bdf2=pd.read_csv(f2)


# In[ ]:

bdf1.head()


# In[ ]:

bdf1.columns


# In[ ]:

bdf2.head()


# In[ ]:



