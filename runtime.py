from soundDataBase import EscSoundDataBase
from audioProcessingUtil import AudioProcessing
from imageProcessingUtilsub import ImageProcessing
from soundClassifier import SpectrogramClassifier
from audio import Audio
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import pyaudio
from time import time
import librosa
import os
import sys
import wave
import soundfile as sf
from gi.repository import Gtk
import easygui

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":

    db_name = 'live'
    db_src = '/home/drazel/Desktop/Env_sound/envSoundClass'
    db_file_type = 'wav'

    live = EscSoundDataBase(db_name=db_name, db_src=db_src, db_file_type=db_file_type)
    
    features_train = None
    features_test = None



    clf = SpectrogramClassifier(live)
    
    #classes
    target_names = ['Dog bark','Rain','Sea waves','Baby Cry','Clock tick',
                    'Person sneeze','Helicopter','Chainsaw','Rooster','Fire Crackling']
    
    '''
    #fresh extraction
    features = clf.extractFeatures(fs=24000, n_fft=512, win_length=480, hop_length=120,
                                   spec_range=(0, 255), spec_pixel_type=np.uint8, spec_log_amplitude=True,
                                   spec_label_range=(0, 255), spec_label_pixel_type=np.uint8,
                                   spec_label_log_amplitude=True,
                                   initial_labels=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250], no_labels=2,
                                   histogram_bins=np.arange(256), histogram_density=True,
                                   glcm_distances=[3, 5], glcm_angles=[0, np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0],
                                   n_mfcc=60, lpc_order=25,
                                   max_segments_per_file=1)
    #print(features)
    clf.save_features('Features/live_test.xlsx')
    '''

    clf_list = [r'Features/live_test.xlsx']


    df = pd.read_excel(r'Features/esc_10_main.xlsx')
    try:

        df = df.drop('FirstOrder_Maximum_0',axis = 1)
        df = df.drop('FirstOrder_Range_0',axis = 1)

        df = df.drop('FirstOrder_Maximum_1',axis = 1)
        df = df.drop('FirstOrder_Range_1',axis = 1)

        df = df.drop('FirstOrder_Maximum_2',axis = 1)
        df = df.drop('FirstOrder_Range_2',axis = 1)

        df = df.drop('FirstOrder_Maximum_3',axis = 1)
        df = df.drop('FirstOrder_Range_3',axis = 1)

    except:
        print('except in SC')

    group_df = df.drop('FileName',axis = 1)
            # group_df = group_df.drop('ClassLabel',axis = 1)
    #group_df = group_df.drop('ValidationNo',axis = 1)

    g1 = group_df.as_matrix()
            #print('g=',g)
    f1 = group_df.drop('ClassLabel',axis=1).as_matrix()
    l1 = g1[:,0]
            #splitting for my work --LA
    for temp in range(len(l1)):            
        l1[temp] = int(l1[temp].split('/')[-1])
            #print('f=',f)

    features_train = f1
    training_labels = l1

    

    normalizer = MinMaxScaler()
    normalizer.fit(features_train)
    training_samples = normalizer.transform(features_train)
    #--------------------------------------------------------------------------->>>>copy features_test later
    
    parameters = {'kernel':('linear','rbf','poly','sigmoid'), 'C':(1, 10,100,1000,10000)}    
    clf = GridSearchCV(SVC(probability=False), parameters)
    #print("******************CLF ",clf)
    training_labels = training_labels.astype('int')
    clf.fit(training_samples,training_labels)




    fs = 24000
    n_fft = 512
    win_length = 480
    hop_length = 120
    spec_range = (0,255)
    spec_pixel_type = np.uint8
    spec_log_amplitude = True
    spec_label_range = (0,255)
    spec_label_pixel_type = np.uint8
    spec_label_log_amplitude = True
    initial_labels = [25,50,75,100,125,150,175,200,225,250]
    no_labels = 2
    histogram_bins = np.arange(256)
    histogram_density = True
    glcm_distances = [3,5]
    glcm_angles = [0,np.pi/4.0,np.pi/2.0, 3*np.pi/4.0]
    n_mfcc = 60
    lpc_order = 25
    max_segments_per_file = 1
    features = None

        #commented to avoid repeated database file generation

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5

    print('features train',features_train)
    for k in range(5):
        aud = pyaudio.PyAudio()
        
            # start Recording
        stream = aud.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        print("recording...")
        frames = []
        
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("finished recording")
      
        
            # stop Recording
        stream.stop_stream()
        stream.close()
            #audio.terminate()
             
        file = "file"+str(k)+".wav"
        waveFile = wave.open(file, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(aud.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()            
        
        feature_columns = ['FileName','ClassLabel','SegmentLabel']
            # resampling segment to 24000 khz
        audio = Audio("/home/drazel/Desktop/Env_sound/envSoundClass/"+file,sampling_rate=fs)

                # removing low frequence components below 500Hz
        data = AudioProcessing.butter_highpass_filter(audio.data,500,fs)

                # Applying k-means and obtianing labels from the spectrogram
        spec_labels = AudioProcessing.get_spectrogram_label(data,n_fft=n_fft,
                                                          win_length=win_length,
                                                          hop_length=hop_length,
                                                          range=spec_label_range,
                                                          pixel_type = spec_label_pixel_type,
                                                          log_amplitude=spec_label_log_amplitude,
                                                          initial_labels=initial_labels,
                                                          no_labels=no_labels)

                # obtaining segments from audio file using the labels obtained from spectrogram segmentation
        segments = AudioProcessing.segmentAudioBySpectrograms(data,spec_labels,win_length,hop_length,max_segments=max_segments_per_file)

                # extracting different features
        #print("Extracting features..")
        j = 0

                # iterating through all segments to obtain features
        for segment in segments:
            seg_data = data[segment[0]:segment[1]]

                # obtaining spectrogram with higher resolution of the segment
            seg_spec = AudioProcessing.get_spectrogram(seg_data,n_fft=2*n_fft,win_length=win_length,hop_length=int(hop_length/2),range=spec_range,pixel_type = spec_pixel_type,log_amplitude=spec_log_amplitude)

                    # median filtering with radius - 3
            med = ImageProcessing.median_image_filter(seg_spec,radius=(3,3,3))

            general_info = np.column_stack([file,5,j])

                    # first order statistics of the spectrogram image
            pixelFeatures = ImageProcessing.getPixelFeatureVector(med,histogram_bins=histogram_bins,histogram_density = histogram_density)

                    # extracting acoustic features
            audio_features = AudioProcessing.get_audio_features(seg_data,fs,n_fft,hop_length,n_mfcc)
            audio_features_mean = np.mean(audio_features,axis=0)
            audio_features_mean = np.column_stack(audio_features_mean)
            audio_features_var = np.var(audio_features,axis=0)
            audio_features_var = np.column_stack(audio_features_var)

                    # obtaining glcm features
            glcmFeatures = ImageProcessing.getGLCMFeatureVector(med,distances=glcm_distances,angles=glcm_angles)
                    # concatenating all the features to obtain feature vector of a segment
            singleSegFeatures = np.concatenate((general_info,pixelFeatures,audio_features_mean,audio_features_var,glcmFeatures),axis = 1)


            features = singleSegFeatures
            #features = np.concatenate((features,singleSegFeatures))

            j = j + 1


        # forming header columns for the dataframe
        pixelColumns = ImageProcessing.getPixelFeatureVectorColumns()

        glcmColumns = ImageProcessing.getGLCMColumnNames(distances=glcm_distances,angles=glcm_angles)

        feature_columns.extend(pixelColumns)
        audio_feature_mean_columns = AudioProcessing.get_audio_feature_columns(n_mfcc,'mean')
        audio_feature_var_columns = AudioProcessing.get_audio_feature_columns(n_mfcc,'var')
        feature_columns.extend(audio_feature_mean_columns)
        feature_columns.extend(audio_feature_var_columns)
        feature_columns.extend(glcmColumns)
 
        df2 = pd.DataFrame(features,columns=feature_columns)
        df2.to_excel('Features/live.xlsx')
        '''
        try:

            df2 = df2.drop('FirstOrder_Maximum_0',axis = 1)
            df2 = df2.drop('FirstOrder_Range_0',axis = 1)

            df2 = df2.drop('FirstOrder_Maximum_1',axis = 1)
            df2 = df2.drop('FirstOrder_Range_1',axis = 1)

            df2 = df2.drop('FirstOrder_Maximum_2',axis = 1)
            df2 = df2.drop('FirstOrder_Range_2',axis = 1)

            df2 = df2.drop('FirstOrder_Maximum_3',axis = 1)
            df2 = df2.drop('FirstOrder_Range_3',axis = 1)

        except:
            print('except in SC')
        '''
        df2 = df2.drop('FileName',axis = 1)
        df2 = df2.drop('ClassLabel',axis = 1)
        df2 = df2.drop('SegmentLabel',axis=1)
        #df2 = df2.drop('ValidationNo',axis = 1)

        gpp = df2.as_matrix()
        features_test = gpp
        #print('featues_test', features_test)
        test_samples = normalizer.transform(features_test)
        pred = clf.predict(test_samples)
        print('pred=',pred)
        try:
            print('class=',target_names[pred[0]-1])
            #gtk_window_set_transient_for()
            dialog = Gtk.MessageDialog(None, 0, Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK, "ALERT")
            dialog.format_secondary_text(target_names[pred[0]-1])
            dialog.run()
            
        except:
            print('class print failure, check pred')



    aud.terminate()
