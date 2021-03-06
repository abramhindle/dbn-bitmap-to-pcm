import theanets
import pickle
import numpy as np
import climate
import logging
import os

climate.enable_default_logging()

# input 64*64 grayscale bitmap
# output samples 22050/30
# fft windows of 1024
# cut down to real values
# cut down again
inputs = 4096
win_size = 2048
swin_size = win_size / 2 + 1
output_size = swin_size * 2
hidlayersize = output_size #win_size
exp = theanets.Experiment(theanets.Regressor,layers=[4096, hidlayersize, hidlayersize, hidlayersize, output_size])
net = exp.network
if os.path.isfile("current_brain.pkl"):
    logging.info("loading existing brain")
    net = pickle.load(file("current_brain.pkl"))
    net._functions = {}
    net._graphs = {}
#elif os.path.isfile("current_pre_brain.pkl"):
#    logging.info("loading existing pre brain")
#    net = pickle.load(file("current_pre_brain.pkl"))
#    net._functions = {}
#    net._graphs = {}

logging.info("Read frames.pkl")
frames = pickle.load(file('fft-frames.pkl'))
logging.info("Read stft.pkl")
audio  = pickle.load(file('stft.pkl'))
train = frames
outputs = audio
train = train.astype(np.float32)
outputs = outputs.astype(np.float32)[0:train.shape[0]]
shuffleids = np.arange(train.shape[0])
np.random.shuffle(shuffleids)
train = train[shuffleids]
outputs = outputs[shuffleids]
i = 0
logging.info("Pretraining")
#for traint, validt in net.train([train, outputs], 
net.train([train, outputs], 
          learning_rate=1e-3,
          save_progress="current_pre_brain.pkl",
          save_every=4,
          batch_size=100,
          train_batches=100,
          num_updates=100,
          patience = 1,
          min_improvement = 0.1,
          algo='layerwise',
          momentum=0.9)
#    print('pretrain i ',str(i))
#    print('training loss:', traint['loss'])
#    print('most recent validation loss:', validt['loss'])
#    print('training err:', traint['err'])
#    print('most recent validation err:', validt['err'])
#    i += 1

i = 0
for traint, validt in net.itertrain([train, outputs], 
          algo='nag',
          learning_rate=1e-3,
          save_progress="current_brain.pkl",
          save_every=4,
          batch_size=100,
          momentum=0.9):
    print('i ',str(i))
    print('training loss:', traint['loss'])
    print('most recent validation loss:', validt['loss'])
    print('training err:', traint['err'])
    print('most recent validation err:', validt['err'])
    i += 1

net.save('stft-theanet.py.net.pkl')


