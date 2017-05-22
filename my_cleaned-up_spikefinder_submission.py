import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import zipfile
import StringIO
import scipy
import keras
import keras.backend as K
from sys import path

path.append('spikefinder-python/')
path.append('OASIS/')
from spikefinder import score
from oasis import oasisAR2
from functions import estimate_parameters


# grab and unzip data
for foo in ('train', 'test'):
    r = requests.get('https://s3.amazonaws.com/neuro.datasets/challenges/spikefinder/' +
                     'spikefinder.' + foo + '.zip', stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall()


# parameters for each dataset obtained by Bayesian optimization / parameter sweep
# [decay(kernel), rise(kernel), percentile(background), window_length(background),
# shift(sparsity), noise_multiplier(sparsity)]
params = np.array([[0.992799, 0.042201, 26, 5000, 45.5894, -0.564023],
                   [0.991437, 0.295501, 20, 5000, 12.7123, -0.0696729],
                   [0.996053, 0.168118, 20, 5000, 8.72845, -0.441365],
                   [0.985170, 0.013046, 20, 5000, 13.8540, 0.0334399],
                   [0.994729, 0.886275, 21, 5000, 22.4001, -0.550495],
                   [0.988732, 0.584366, 28, 4000, 4.15617, -0.238822],
                   [0.967307, 0.474290, 29, 4000, 0.07804, 0.222812],
                   [0.992499, 0.795482, 23, 5000, 20.6628, -0.232531],
                   [0.993040, 0.141836, 22, 4000, 41.1818, 0.00033394],
                   [0.975445, 0.531893, 27, 4000, 45.6739, -0.352258]])

# run OASIS for dataset with provided (uglily dataset dependent) parameters
def runOASIS(dataset, d, r, perc, window, lam_sn, mu, train=True):
    foo = 'train' if train else 'test'
    Y = np.array(pd.read_csv('spikefinder.' + foo + '/' +
                             str(dataset) + '.' + foo + '.calcium.csv')).T
    prep = [y[~np.isnan(y)] - scipy.ndimage.filters.percentile_filter(
        y[~np.isnan(y)], perc, window) for y in Y]
    S = np.nan * np.zeros_like(Y)
    for i, y in enumerate(Y):
        preprocessedy = prep[i]
        y = y[~np.isnan(y)]
        # decimate to estimate noise (upsampling of spikefinder data produced artefacts)
        ydec = y[:len(y) // 10 * 10].reshape(-1, 10).mean(1)
        g, sn = estimate_parameters(ydec, 1, lags=10, fudge_factor=.97, method='mean')
        S[i, :len(preprocessedy)] = oasisAR2(
            preprocessedy - mu, d + r, -d * r, lam=lam_sn * sn *
            np.sqrt((1 + d * r) / ((1 - d * d) * (1 - r * r) * (1 - d * r))))[1]
    return S


# define pearson correlation loss
def pearson_loss(x, y):
    x_ = x - K.mean(x, axis=1, keepdims=True)
    y_ = y - K.mean(y, axis=1, keepdims=True)
    corr = K.sum(x_ * y_, axis=1) / K.sqrt(K.sum(K.square(x_), axis=1) *
                                           K.sum(K.square(y_), axis=1) + 1e-12)
    return -corr


# Use Keras to build a simple shallow 1 layer NN
def make_decoder(w, optimizer='adadelta'):
    F = keras.layers.Input(shape=(None, 1))
    s = keras.layers.Convolution1D(1, len(w), activation='relu', border_mode='same',
                                   weights=[w[:, None, None], np.array([0])])(F)
    decoder = keras.models.Model(input=F, output=s)
    decoder.compile(optimizer=optimizer, loss=pearson_loss)
    return decoder


def run(dataset=1, tau=30, b=False, verbose=0, optimizer='adadelta'):
    trueS = np.array(pd.read_csv('spikefinder.train/' + str(dataset) + '.train.spikes.csv')).T
    # RUN OASIS
    infS = runOASIS(dataset, *params[dataset - 1])
    cor0 = np.mean(np.nan_to_num(score(pd.DataFrame(infS.T), pd.DataFrame(trueS.T))))
    # POSTPROCESS by convolving result with some kernel
    # solve regression problem to obtain kernel
    k = []
    for n in range(len(trueS)):
        s = trueS[n]
        s = np.hstack([np.zeros(tau // 2), s[~np.isnan(s)]])
        T = len(s)
        infs = np.hstack([infS[n], np.zeros(tau // 2)])[:T]
        ss = np.zeros((tau, T))
        for i in range(tau):
            ss[i, i:] = infs[:T - i]
        ssm = ss - ss.mean() if b else ss
        k.append(np.linalg.lstsq(np.nan_to_num(ssm.T), s)[0])
    km = np.mean(k, 0)
    k = [kk * km.dot(kk) / kk.dot(kk) for kk in k]  # normalize
    km = np.mean(k, 0)
    # convolve to obtain result
    S = np.clip(np.array([np.convolve(s, km)[tau // 2:-tau // 2 + 1]
                          for i, s in enumerate(infS)]), 0, np.inf)
    cor1 = np.mean(np.nan_to_num(score(pd.DataFrame(S.T), pd.DataFrame(trueS.T))))
    # REFINE postprocessing kernel by optimizing for correlation instead of least squares
    decoder = make_decoder(km[::-1], optimizer)  # initialize weights with least-squares solution
    history = decoder.fit(np.nan_to_num(infS[..., None]), np.nan_to_num(trueS[..., None]),
                          batch_size=len(trueS), epochs=100, verbose=verbose)
    w, b = history.model.get_weights()
    pred = np.squeeze(decoder.predict(np.nan_to_num(infS[..., None]).astype('float32')))
    cor2 = np.mean(np.nan_to_num(score(pd.DataFrame(pred.T), pd.DataFrame(trueS.T))))
    if verbose:
        print cor0, cor1, cor2
        plt.plot(km)
        plt.plot(np.ravel(w))
        plt.show()
    if cor1 > cor2:  # save whichever is better
        pd.DataFrame(S.T).to_csv('spikefinder.train/' +
                                 str(dataset) + '.train.predict.csv', index=False)
    else:
        pd.DataFrame(pred.T).to_csv('spikefinder.train/' +
                                    str(dataset) + '.train.predict.csv', index=False)
    if dataset < 6:  # run also on test data
        infStest = runOASIS(dataset, *params[dataset - 1], train=False)
        if cor1 > cor2:
            Stest = np.clip(np.array([np.convolve(s, km)[tau // 2:-tau // 2 + 1]
                                      for i, s in enumerate(infStest)]), 0, np.inf)
        else:
            Stest = np.squeeze(decoder.predict(
                np.nan_to_num(infStest[..., None]).astype('float32')))
        pd.DataFrame(Stest.T).to_csv('spikefinder.test/' +
                                     str(dataset) + '.test.predict.csv', index=False)


# run for each dataset
cor = []
for d in range(1, 11):
    print 'Dataset ', d
    run(d)
    sc = score(pd.read_csv('spikefinder.train/' + str(d) + '.train.spikes.csv'),
               pd.read_csv('spikefinder.train/' + str(d) + '.train.predict.csv'))
    print np.mean(sc), np.median(sc)
