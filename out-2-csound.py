import wave
import random

nblock = 735
wavefile = wave.open("out.wav",'r')
nframes = wavefile.getnframes()
rate = wavefile.getframerate()
nchannels = wavefile.getnchannels()
duration = (nframes / nchannels) / float(rate)

def linscale(x,mins,maxs):
	return x*(maxs - mins) + mins

def frames2time(frames):
	return frames / float(rate)

def grain(when, duration, amp, curframe,length):
    return 'i"grain" %s %s %s %s %s %s' % (when, duration,amp,curframe,length, curframe/float(nframes))

def pgrain(when, duration, amp, curframe,length):
    print grain(when, duration, amp, curframe,length)

print ('f 101 0 %s 1 "%s" 0 4 1' % (nframes, "out.wav"))
curframes = 0
onesamp = 1/float(rate)
while(curframes < nframes):
    ntimes = int(linscale(random.random(),1,20))
    for j in xrange(0,ntimes):
        when = curframes + int(linscale(random.random(),-1*nblock, (j+1)*nblock))
        # print parent 
        pgrain(frames2time(when), frames2time(nblock),1.0,curframes,nblock)
        children = int(linscale(random.random(),1,20))
        cdur = nblock
        for i in xrange(0,children):
            if (cdur > 100):
                cwhen = when + int(linscale(random.random(),0,nblock*(rate/1000)))
                cdur = int(cdur * linscale(random.random(),0.25,0.99))
                print "; %s child" % i
                pgrain(frames2time(cwhen), frames2time(cdur),1.0/(i+1),curframes,cdur)
    curframes += nblock
