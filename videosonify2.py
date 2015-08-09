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
brain = theanets.feedforward.Regressor.load("theanet.py.net.pkl")#brain-1438666035")
#brain = theanets.feedforward.Regressor.load("brain-1438666035")
brain._graphs = {} 
brain._functions = {}
outwav = scikits.audiolab.Sndfile("out.wav",mode='w',format=scikits.audiolab.Format(),channels=1,samplerate=22050)
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
alen = 735 # audio length
window = np.hanning(alen)
frames = 0
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
    out = brain.predict([scaled,gaussian_noise(scaled,0.0,0.1),gaussian_noise(scaled,-0.05,0.1)])
    out *= 2.0
    out -= 1.0

    #out = out.transpose()
    #z = np.zeros(735*2)
    #z[0:735] = out[0]
    #z[735:] = out[0]
    #sd.play(z.transpose(), 44100)
    #sd.wait()
    out[0] *= window
    out[1] *= window
    out[2] *= window
    outs.append(out)
    outwav.write_frames(out[0])
    #k = cv2.waitKey(1) & 0xff
    #if k == 27:
    #    continue
    frames += 1
    if frames % 30 == 0:
        print frames

outwav.sync()

outwav = scikits.audiolab.Sndfile("wout.wav",mode='w',format=scikits.audiolab.Format(),channels=1,samplerate=22050)
output = np.zeros(735*(2+len(outs)))
for i in range(0,len(outs)):
    #audio = outs[i]*window
    start = (i + 1)*alen
    end = start + alen
    rstart = start + alen/2 + (random.random() - 0.5) * (alen/10) #int(start - (alen/2) + alen*random.random())
    rend = rstart + alen
    output[start:end] += outs[i][0]
    output[rstart:rend] += outs[i][1]
    output[(rstart-alen):(rend-alen)] += outs[i][1]

outwav.write_frames(output)
outwav.sync()

cv2.destroyAllWindows()
