#!/home/a/anaconda3/envs/tinysleepnet/bin/python

#!/bin/env python

import argparse
import glob
import importlib
import os
import numpy as np
import shutil
import sklearn.metrics as skmetrics
import tensorflow as tf


''' todo:
    1) look around for a sleep dataset with electrodes similar to muse
    2) add additional data for each epoch:
        spectrogram?
        respiratory rate?
        spo2
        (noise, light, spo2, pulse...)
    3) generate a daily report notebook
    4) questions:
        -why is AF7 and AF8 poor at predicting rems
        -why am I not registering much deep sleep?
        -how well can TP9 and TP10 substitute for Cz and Pz
    '''



from data import load_data, get_subject_files
from model import TinySleepNet
from minibatching import (iterate_minibatches,
                          iterate_batch_seq_minibatches,
                          iterate_batch_multiple_seq_minibatches)
from utils import (get_balance_class_oversample,
                   print_n_samples_each_class,
                   save_seq_ids,
                   load_seq_ids)
from logger import get_logger

import scriptine


config = {
    # Train
    "n_epochs": 200,
    "learning_rate": 1e-4,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "clip_grad_value": 5.0,
    "evaluate_span": 50,
    "checkpoint_span": 50,

    # Early-stopping
    "no_improve_epochs": 50,

    # Model
    "model": "model-mod-8",
    "n_rnn_layers": 1,
    "n_rnn_units": 128,
    "sampling_rate": 100.0,
    "input_size": 3000,
    "n_classes": 5,
    "l2_weight_decay": 1e-3,

    # Dataset
    "dataset": "sleepedf",
    "data_dir": "./data/sleepedf/sleep-cassette/eeg_fpz_cz",
    "n_folds": 20,
    "n_subjects": 20,

    # Data Augmentation
    "augment_seq": True,
    "augment_signal_full": True,
    "weighted_cross_ent": True,
}

config.update({
    "seq_length": 20,
    "batch_size": 15,
})

config.update({
    "batch_size": 1,
    "seq_length": 1,
})
config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)


import pandas as pd


def getAccData(h5fname):
    x = pd.read_hdf(h5fname, 'Muse/ACC').reset_index(drop=True)
    x = x[x.ts.cummax().diff()>0]
    x['orientation'] = 180/np.pi*np.arctan(x.Y/x.X)
    x['motion'] = x.Y.rolling(1000).std().fillna(0)
    return x

def getPpgData(h5fname):
    #read the data
    x = pd.read_hdf(h5fname, 'Muse/PPG').reset_index(drop=True)
    x = x[x.ts.cummax().diff()>0]
    x.index = x.ts

    #label with heartbeat cycle (based on PPG1)
    x['PPG1m'] = x.PPG1.rolling(5).mean()
    isPeak = x['PPG1m'] == x['PPG1m'].rolling(int(.5*64), center=True).max()
    isPeak -= isPeak.rolling(20).max().shift(1).fillna(0)    #remove consecutive isPeak trues
    isPeak = isPeak.clip(0,1)
    x['cycle'] = isPeak.cumsum()

    #compute PPG1 stats on each cycle
    b = x['PPG1m'].groupby(x['cycle']).agg(['count','min','first','last','idxmin', ('idxfirst',lambda y: y.index.values[0]), ('idxlast',lambda y: y.index.values[-1])])
    b['max'] = b['first'] + (b['last']-b['first']) * (b['idxmin']-b['idxfirst']) / (b['idxlast']-b['idxfirst'])
    ppg = pd.DataFrame()
    ppg['ts'] = b['idxfirst']
    ppg['bpm'] = 1 / (b['count']/64/60)
    ppg['ampl1'] = (b['max']-b['min']) / (b['max']+b['min'])

    #compute PPG2 stats on each cycle
    x['PPG2m'] = x.PPG2.rolling(5).mean()
    b = x['PPG2m'].groupby(x['cycle']).agg(['count','min','first','last','idxmin', ('idxfirst',lambda y: y.index.values[0]), ('idxlast',lambda y: y.index.values[-1])])
    b['max'] = b['first'] + (b['last']-b['first']) * (b['idxmin']-b['idxfirst']) / (b['idxlast']-b['idxfirst'])
    ppg['ampl2'] = (b['max']-b['min']) / (b['max']+b['min'])
    ppg['spo2'] = 0
    return ppg

def predict_command(h5fname):
    # Add dummy class weights
    trues = []
    preds = []
    #tf.random.set_random_seed(123)

    model = TinySleepNet(
        config=config,
        output_dir='out_sleepedf/train/0',
        use_rnn=True,
        testing=True,
        use_best=True,
    )

    x = pd.read_hdf(h5fname, 'Muse/EEG').reset_index(drop=True)

    #resample to 100Hz
    x = x[x.ts==x.ts.cummax()]
    ts = np.arange(x.ts.values[0], x.ts.values[-1], 10**7)
    ts -= ts%1000
    x = x.set_index('ts').reindex(ts, method='nearest')

    #
    n_epochs = len(x)//3000
    x = x.iloc[:n_epochs*3000]
    p = pd.DataFrame()
    p['ts'] = x.index[::3000]
    for c in 'AF7 AF8 TP9 TP10'.split():
        night_x = x[c].values.reshape( (n_epochs,3000,1,1))
        print('night_x', night_x.shape, night_x.mean(), night_x.std())
        night_y = np.zeros(n_epochs)


        # Create minibatches for testing
        test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
            [night_x],
            [night_y],
            batch_size=config["batch_size"],
            seq_length=config["seq_length"],
            shuffle_idx=None,
            augment_seq=False,
        )

        # Evaluate
        test_outs = model.evaluate(test_minibatch_fn)
        #from IPython import embed; embed()
        print('test_outs', c, np.array(test_outs['test/preds']).mean())
        p[c] = test_outs['test/preds']

        # Filter bad data
        isBad = night_x[:,:,0,0].std(axis=1) > 500
        print(c, isBad.sum(), p[c].mean(), p[c].std())
        p[c][isBad] = -1

    #acc data
    y = getAccData(h5fname)
    y = y.set_index('ts').reindex(p.ts, method='ffill')
    for c in ['orientation','motion']:
        p[c] = y[c].values

    #ppg data
    ppg = getPpgData(h5fname)

    #save results
    store = pd.HDFStore(h5fname, complib='zlib',complevel=5)
    store.put('tinysleepnet', p)
    store.put('spo2', ppg)

    #from IPython import embed; embed()
    tf.reset_default_graph()



if __name__ == "__main__":
    scriptine.run()
    