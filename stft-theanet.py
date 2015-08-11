import theanets
import pickle
import numpy as np
import climate
import logging

climate.enable_default_logging()

# input 64*64 grayscale bitmap
# output samples 22050/30
# fft windows of 1024
# cut down to real values
# cut down again
inputs = 4096
win_size = 1024
swin_size = win_size / 2 + 1
output_size = swin_size
hidlayersize = win_size
exp = theanets.Experiment(theanets.Regressor,layers=[4096, hidlayersize, hidlayersize, hidlayersize, output_size])
net = exp.network

logging.info("Read frames.pkl")
frames = pickle.load(file('frames.pkl'))
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
for traint, validt in net.itertrain([train, outputs], 
          algo='nag',
          learning_rate=1e-4,
          save_progress="current_brain.pkl",
          save_every=10,
          batch_size=100,
          momentum=0.9):
    print('i ',str(i))
    print('training loss:', traint['loss'])
    print('most recent validation loss:', validt['loss'])
    print('training err:', traint['err'])
    print('most recent validation err:', validt['err'])
    i += 1

net.save('stft-theanet.py.net.pkl')


