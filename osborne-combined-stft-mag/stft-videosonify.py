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
# outwav.write_frames(windowed)

last_phase = np.zeros(window_size)
invwindow = 1.0/scipy.hamming(window_size)
amax=7e-3

# Exp001: init all phase @ pi/2 + static phase of np.pi/100.0 windowed
#       [X] 30hz :( [ ] aesthetic [X] Robot
# Exp002: init all phase @ pi/2 + static phase of np.pi/10.0 windowed
#       [X] 30hz :( [ ] aesthetic [X] Robot
# Exp003: init all phase @ pi/2 + static phase of var np.pi/2.0 normally distributed windows
#       [X] 30hz :( [X] aesthetic [ ] Robot [X] Pulsing
# Exp004: init all phase @ pi/2 + static phase of var np.pi/10.0 normally distributed windows
#       [X] 30hz :( [ ] aesthetic [X] Robot [ ] Pulsing
# Exp005: init all phase @ pi/2 + static phase of var np.pi normally distributed windows
#       [X] 30hz :( [ ] aesthetic [X] Robot [ ] Pulsing [X] White Noisey
# Exp006: init all phase @ 0 + static phase of var np.pi/2.0 normally distributed windows
#       [X] 30hz :( [ ] aesthetic [ ] Robot [X] Pulsing
# Exp007: init all phase @ pi/2 + static phase of var np.pi/2.0 uniformly distributed windows
#       [X] 30hz :( [ ] aesthetic [ ] Robot [X] Pulsing
#       more noisey
# Exp008: init all phase @ pi/2 + static phase of 0 to pi/2
#       [X] 30hz :( [ ] aesthetic [ ] Robot [ ] Pulsing
# Exp009: init all phase @ pi/2 + static phase of pi/2 to 0
#       [X] 30hz :( [ ] aesthetic [ ] Robot [ ] Pulsing
# Exp010: init normals -pi/2 to pi/2 + static phase of pi/2
#       [X] 30hz :( [ ] aesthetic [X] Robot [ ] Pulsing
# Exp011: init normals -pi/2 to pi/2 + random_normals pi/10 recursive
#       [ ] 30hz :( [ ] aesthetic [X] Robot [ ] Pulsing

cones = np.zeros(swin_size-1).astype(complex) + complex(0,1)
oldout = np.zeros(swin_size)

phase = np.random.normal(0,np.pi/2, window_size)
# phase       = np.random.normal(np.pi/2,np.pi,window_size)
# staticphase = np.random.normal(0,np.pi/2.0,window_size)
staticphase = np.ones(window_size).astype(float32)*np.pi/2.0

#phase = np.zeros(window_size)

dooverlaps = True
dowindowing = True

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

    out = brain.predict([scaled])[0]

    # out is the guts of a fourier transform
    # inverse fft won't work well
    buf = np.zeros(window_size).astype(complex)
    buf[0:swin_size] += out[0:swin_size]

    # mirror around
    # buf[swin_size:window_size] += -1*buf[1:swin_size-1][::-1]
    
    # make phase
    phase += np.random.normal(0,np.pi/10,window_size)
    myfft = buf * exp(complex(0,1) * phase)
    
    audio = scipy.real(scipy.ifft(myfft))
    if (dowindowing):
        audio *= windowed 
        
    last_phase = np.angle(scipy.fft(audio))

    amax = max(audio.max(), amax)

    if (dooverlaps):
        audio[0:olaps] += overlap[0:olaps]
        ## should be a copy but whatever
        overlap[0:olaps] *= 0
        overlap[0:olaps] += audio[window_size-olaps:window_size]
        outwav.write_frames(audio[0:olaps]/amax)
    else:
        outwav.write_frames(audio/amax)
        #outwav.write_frames(windowed)

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
