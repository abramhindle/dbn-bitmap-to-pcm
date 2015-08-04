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
from scikits.audiolab import play
import theanets
import pickle
import numpy as np
import scikits.audiolab
if len(sys.argv) < 2:
    print "Opening vtest.avi"
    cap = cv2.VideoCapture("vtest.avi")
else:
    print "Opening %s" % sys.argv[1]
    cap = cv2.VideoCapture(sys.argv[1])

running = True

frames = []

# load brain

brain = theanets.feedforward.Regressor.load("theanet.py.net.pkl")#brain-1438666035")
#brain = theanets.feedforward.Regressor.load("brain-1438666035")
brain._graphs = {} 
brain._functions = {}
outwav = scikits.audiolab.Sndfile("out.wav",mode='w',format=scikits.audiolab.Format(),channels=1,samplerate=22050)
while(running):
    ret, frame = cap.read()
    if (not ret):
        running = False
        continue
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(grey, (64,64))
    scaled = scaled.astype(np.float32)
    scaled /= 255.0
    scaled = scaled.flatten()
    out = brain.predict([scaled])
    out *= 2.0
    out -= 1.0
    out.transpose()
    #play(out[0], fs=22050)
    outwav.write_frames(out[0])

outwav.sync()
