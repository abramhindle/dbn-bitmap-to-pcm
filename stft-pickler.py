import sys
import cv2
import numpy as np
from numpy import *
import random
import time
import pickle
import os.path
import scipy.io
import scipy.io.wavfile

# run me to generate frames.pkl and audio.pkl

if len(sys.argv) < 2:
    print "Opening vtest.avi"
    cap = cv2.VideoCapture("vtest.avi")
else:
    print "Opening %s" % sys.argv[1]
    cap = cv2.VideoCapture(sys.argv[1])

running = True

frames = []

while(running):
    ret, frame = cap.read()
    if (not ret):
        running = False
        continue
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(grey, (64,64))
    scaled = scaled.astype(np.float32)
    scaled /= 255.0
    frames.append( scaled.flatten() )
        
pickle.dump(np.array(frames), file('frames.pkl','wb'))

if not os.path.isfile("vtest.wav"):
    os.system("avconv -i vtest.avi  -acodec pcm_s16le -ar 22050 -ac 1  vtest.wav")


samples = 735
window_size = 1024
swin_size = window_size / 2 + 1
windowed = scipy.hanning(window_size)

wav = scipy.io.wavfile.read("vtest.wav")
wavdata = wav[1].astype(np.float32)
norm = (wavdata)/(2.0**15)
# pad norm with zeros
nsamples = int(math.ceil(len(norm)/float(samples)))
norm.resize(samples*nsamples)
norm.resize(window_size*math.ceil(len(norm)/float(window_size)))
# we're dumping phase 
ffts = np.array([scipy.real(scipy.fft(norm[i*samples:i*samples+window_size] * windowed))[0:swin_size] for i in range(0,nsamples)])
ffts = (ffts + ffts.min())/ffts.max()
pickle.dump(ffts, file('stft.pkl','wb'))
