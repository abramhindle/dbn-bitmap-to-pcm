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
#import sounddevice as sd
import random
#sd.default.samplerate = 44100
#sd.default.channels = 1
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, imshow

if len(sys.argv) < 2:
    print "Opening vtest.avi"
    cap = cv2.VideoCapture("vtest.avi")
else:
    print "Opening %s" % sys.argv[1]
    cap = cv2.VideoCapture(sys.argv[1])

running = True

frames = []

# load brain
# cv2.namedWindow("frame", 1)
brain = theanets.feedforward.Regressor.load("stft-theanet.py.net.pkl")#brain-1438666035")
#brain = theanets.feedforward.Regressor.load("brain-1438666035")
brain._graphs = {} 
brain._functions = {}
outwav = scikits.audiolab.Sndfile("out.wav",mode='w',format=scikits.audiolab.Format(),channels=1,samplerate=30720)
ret, frame = cap.read()

#class BufferPlayer:
#    def __init__(self):
#        self.base = 4096
#        self.size = 2*self.base
#        self.buffer = bp.zeros(self.base)
#        self.oldbuffs = []
#
#    def add(self, arr):
#        self.oldbuffs.append(arr)
#
#    def play(self):
#        ''' play the next thing '''
#        
#        sd.play(out[0], 22050)
    
def gaussian_noise(inarr,mean=0.0,scale=1.0):    
    noise = np.random.normal(mean,scale,inarr.shape)
    return inarr + noise.reshape(inarr.shape)

outs = []
window_size = 2048
windowed = scipy.hanning(window_size)
swin_size = window_size / 2 + 1
alen = 1024 # audio length
window = np.hanning(alen)
frames = 0
overlapsize = window_size - alen
overlap = np.zeros(overlapsize)

# if we have 1/4 overlap
#   __       __
#  /  \__   /  \__
#     /  \__   /  \__
#        /  \     /  \
#0.5111111111111111111
# 
# flat_window = np.ones(window_size)
# olaps = int(math.ceil((window_size - alen)/2.0))
# flat_window[0:olaps] = np.arange(0,olaps)/float(olaps-1)
# flat_window[window_size-olaps:window_size] = 1.0 - flat_window[0:olaps]


# if we have 1/2 overlap
#
#  /\  /\
#   /\/\/\
#    /\  /\
#0.51111110.5

flat_window = np.ones(window_size)
olaps = int(math.ceil((window_size - alen))) # half
flat_window[0:olaps] = np.arange(0,olaps)
flat_window[olaps:window_size] = np.arange(0,olaps)[::-1]
flat_window /= float(olaps-1)

# debug
outwav.write_frames(windowed)
last_phase = np.zeros(window_size)
invwindow = 1.0/scipy.hamming(window_size)
amax=0.00285

cones = np.zeros(swin_size-1).astype(complex) + complex(0,1)

while(running):
    ret, frame = cap.read()
    if (not ret):
        running = False
        continue
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',frame)    
    scaled = cv2.resize(grey, (64,64))
    scaled = scaled.astype(np.float32)
    scaled /= 255.0
    scaled = scaled.flatten()

    # do 3 predictions
    out = brain.predict([scaled])[0]
    # out is the guts of a fourier transform
    # inverse fft won't work well
    buf = np.zeros(window_size).astype(complex)
    buf[0:swin_size] += out[0:swin_size]

    # add some phase
    #buf[1:swin_size] += cones*last_phase[1:swin_size]
    #buf[1:swin_size] += cones*np.pi
    #buf[1:swin_size] += cones*np.random.uniform(-np.pi/16,np.pi/16,swin_size-1)
    # buf[1:swin_size] += cones*np.random.uniform(-0.02,0.02,swin_size-1)
    # mirror around
    # buf[swin_size:window_size] += buf[1:swin_size-1][::-1]
    # phase on the other size is opposite
    # buf[swin_size:window_size] *= -1*cones[0:swin_size-2]
    
    # audio = scipy.real(scipy.ifft(buf)) * windowed
    audio = scipy.real(scipy.ifft(buf))
    # audio = np.zeros(window_size) * windowed
    #audio = scipy.real(scipy.ifft(buf)) 
    last_phase = scipy.imag(scipy.fft(audio))

    amax = max(audio.max(), amax)
    
    audio[0:olaps] += overlap[0:olaps]
    ## should be a copy but whatever
    overlap[0:olaps] *= 0
    overlap[0:olaps] += audio[window_size-olaps:window_size]
    outwav.write_frames(audio[0:olaps]/amax)

    #outwav.write_frames(audio/amax)
    #outwav.write_frames(np.zeros(2048))

    #k = cv2.waitKey(1) & 0xff
    #if k == 27:
    #    continue
    frames += 1
    if frames % 30 == 0:
        print (frames, amax)

outwav.write_frames(overlap)

#idea
# for each frame generate interpolate spectra
# for each frame run 1024 sinusoids and sum them manually but maintain phase?
# invent a phase vector that we add to each time to ensure that the next window has appropriate phase?


outwav.sync()

# outwav = scikits.audiolab.Sndfile("wout.wav",mode='w',format=scikits.audiolab.Format(),channels=1,samplerate=22050)
# output = np.zeros(735*(2+len(outs)))
# for i in range(0,len(outs)):
#     #audio = outs[i]*window
#     start = (i + 1)*alen
#     end = start + alen
#     rstart = start + alen/2 + (random.random() - 0.5) * (alen/10) #int(start - (alen/2) + alen*random.random())
#     rend = rstart + alen
#     output[start:end] += outs[i][0]
#     output[rstart:rend] += outs[i][1]
#     output[(rstart-alen):(rend-alen)] += outs[i][1]
# 
# outwav.write_frames(output)
# outwav.sync()
# 
cv2.destroyAllWindows()
