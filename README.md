# SPEECH-EMOTION-PROBLEM
SER from .wav files

 ## Feature Extraction  ##
For training a Convolutional neural network, the features are extracted from raw audio files using Mel-frequency cepstral coefficients (MFCCs). MFCC mimics the logarithmic perception of loudness and pitch of human auditory system and tries to eliminate speaker dependent characteristics by excluding the fundamental frequency and their harmonics.

## Non-Uniform data distribution  ##
The train folder, contains .wav files corresponding to five classes, i.e. happy, sad, fear, disgust, neutral. But the number of .wav files the given class is not evenly distributed. So, I have used data augmentation to even disttributed the number of samples for all the classes and to also increase the size of the training set.(around 29k samples)

## Data Augmentation  ##
To generate syntactic data for audio, we can apply noise injection, shifting time, changing pitch and speed. Numpy provides an easy way to handle noise injection and shifting time while librosa (library for Recognition and Organization of Speech and Audio) help to manipulate pitch and speed.

### Noise Injection ###
It simply add some random value into data by using numpy.
### Shifting Time ###
The idea of shifting time is very simple. It just shift audio to left/right with a random second. If shifting audio to left (fast forward) with x seconds, first x seconds will mark as 0 (i.e. silence). 
### Changing Pitch ###
This augmentation is a wrapper of librosa function. It change pitch randomly.
### Changing Speed ###
Same as changing pitch, this augmentation is performed by librosa function. It stretches times series by a fixed rate.

## Notebooks ###
train.ipynb contains the code for training a convolutional neural network, data augmentations, visualizations, checking validation loss and accuracy. test.py contains the code for making predictions by the model and finally model.h5 contains the trained model. (Note: These are not the trained weights.)
