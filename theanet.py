import theanets
import pickle
import numpy as np
# input 64*64 grayscale bitmap
# output samples 22050/30
inputs = 4096
outputs = 735
exp = theanets.Experiment(theanets.Regressor,layers=[4096, 1000, outputs])
net = exp.network
import climate
import logging

climate.enable_default_logging()


# 4096 * n frames
# 735 * n frames
frames = pickle.load(file('frames.pkl'))
audio  = pickle.load(file('audio.pkl'))
train = frames
valid = audio
train = train.astype(np.float32)
valid = valid.astype(np.float32)[0:train.shape[0]]
#exp.train(train, valid, algo='sgd', learning_rate=1e-4, momentum=0.9)
#exp.train([train, valid], algo='sample')
#net.train([train, valid], algo='rprop', patience=10, batch_size=4)
#exp.train([train, valid], algo='sgd')
#exp.train([train,valid],
#          algo='nag',
#          learning_rate=0.1,
#          momentum=0.9):

i = 0
for traint, validt in net.itertrain([train, valid], 
          algo='nag',
          learning_rate=0.05,
          save_progress="brain-{}",
          save_every=100,
          momentum=0.9):
    print('i ',str(i))
    print('training loss:', traint['loss'])
    print('most recent validation loss:', validt['loss'])
    print('training err:', traint['err'])
    print('most recent validation err:', validt['err'])
    i += 1

net.save('theanet.py.net.pkl')
exp.save('theanet.py.exp.pkl')

