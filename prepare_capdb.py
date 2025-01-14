import argparse
import glob
import math
import ntpath
import os
import shutil
import pyedflib
import numpy as np
import pandas as pd

from sleepstage import stage_dict
from logger import get_logger


# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}


def readAnnotations(fname):
    ann_onsets = []
    ann_durations = []
    ann_stages = []
    stage_xlat = dict(  W = 'Sleep stage W',
                        S1 = 'Sleep stage 1',
                        S2 = 'Sleep stage 2',
                        S3 = 'Sleep stage 3',
                        S4 = 'Sleep stage 4',
                        R = 'Sleep stage R',
                        MT = 'Movement time',
                        )
    onset = -30
    with open(fname) as fd:
        for line in fd:
            line = line.strip().split('\t')
            if (len(line) not in [5,6]
                or not (line[2].startswith('SLEEP-') or line[3].startswith('SLEEP-'))): continue
            onset += 30
            #hms = [int(n) for n in line[1].split(':')]
            #ts = 3600*hms[0] + 60*hms[1] + hms[2]
            #if ts < 12*3600: ts += 24*3600
            ann_onsets.append(onset)
            ann_durations = 30
            ann_stages.append( stage_xlat[line[0]])
    #print(f'read {len(ann_onsets)} epochs from file {fname}')
    return ann_onsets, ann_durations, ann_stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/capdb",
                        help="File path to the capdb dataset.")
    parser.add_argument("--output_dir", type=str, default="./data/capdb",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="Fp2-F4",
                        help="Name of the channel in the dataset.")
    parser.add_argument("--log_file", type=str, default="info_ch_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    args.output_dir = os.path.join(args.output_dir, args.select_ch)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    args.log_file = os.path.join(args.output_dir, args.log_file)

    # Create logger
    logger = get_logger(args.log_file, level="info")

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation from EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*.txt"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))

        psg_f = pyedflib.EdfReader(psg_fnames[i])

        start_datetime = psg_f.getStartdatetime()
        logger.info("Start datetime: {}".format(str(start_datetime)))

        file_duration = psg_f.getFileDuration()
        logger.info("File duration: {} sec".format(file_duration))
        epoch_duration = 30 #psg_f.datarecord_duration
        logger.info("Epoch duration: {} sec".format(epoch_duration))

        # Extract signal from the selected channel
        ch_names = psg_f.getSignalLabels()
        ch_samples = psg_f.getNSamples()

        # sum the requested signals
        signals = 0
        try:
            for sc in select_ch.split('+'):
                select_ch_idx = ch_names.index(sc)
                signals = signals + psg_f.readSignal(select_ch_idx)
        except ValueError: continue

        if 0:
            select_ch_idx = -1
            for s in range(psg_f.signals_in_file):
                if ch_names[s] == select_ch:
                    select_ch_idx = s
                    break
            if select_ch_idx == -1:
                print('skipping recording because channel not found')
                continue

        sampling_rate = psg_f.getSampleFrequency(select_ch_idx)
        if sampling_rate!=512:
            print(f'skipping {psg_fnames[i]} because sampling rate is {sampling_rate}')
            continue
        if 1:   #downsample the signal?
            signals = signals[::5]
            sampling_rate /= 5

        n_epoch_samples = int(epoch_duration * sampling_rate)
        n_epochs = len(signals) // n_epoch_samples
        signals = signals[:n_epoch_samples*n_epochs].reshape(-1, n_epoch_samples)


        logger.info("Select channel: {}".format(select_ch))
        logger.info("Select channel samples: {}".format(ch_samples[select_ch_idx]))
        logger.info("Sample rate: {}".format(sampling_rate))

        # Sanity check
        assert len(signals) == n_epochs, f"signal: {signals.shape} != {n_epochs}"

        #from IPython import embed; embed()

        # Generate labels from onset and duration annotation
        labels = []
        total_duration = 0
        ann_onsets, ann_durations, ann_stages = readAnnotations(ann_fnames[i])
        for a in range(len(ann_stages)):
            onset_sec = int(ann_onsets[a])
            duration_sec = 30  #int(ann_durations[a])
            ann_str = "".join(ann_stages[a])

            # Sanity check
            assert onset_sec == total_duration, f'{onset_sec}!={total_duration}'

            # Get label value
            label = ann2label[ann_str]

            # Compute # of epoch for this stage
            if duration_sec % epoch_duration != 0:
                logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
            duration_epoch = int(duration_sec / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label
            labels.append(label_epoch)

            total_duration += duration_sec

            #logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
            #    onset_sec, duration_sec, label, ann_str
            #))
        labels = np.hstack(labels)

        # Remove annotations that are longer than the recorded signals
        labels = labels[:len(signals)]

        # Get epochs and their corresponding labels
        x = signals.astype(np.float32)
        y = labels.astype(np.int32)

        # Select only sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        logger.info("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        logger.info("Data after selection: {}, {}".format(x.shape, y.shape))

        # Remove movement and unknown
        move_idx = np.where(y == stage_dict["MOVE"])[0]
        unk_idx = np.where(y == stage_dict["UNK"])[0]
        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)
            logger.info("Remove irrelevant stages")
            logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
            logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
            logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
            logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
            select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
            x = x[select_idx]
            y = y[select_idx]
            logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))

        # Save
        filename = f'subject{i+1}.npz'
        save_dict = {
            "x": x, 
            "y": y, 
            "fs": sampling_rate,
            "ch_label": select_ch,
            "start_datetime": start_datetime,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,
            "n_all_epochs": n_epochs,
            "n_epochs": len(x),
        }
        print(filename, x.shape, y.shape, sampling_rate)
        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        logger.info("\n=======================================\n")


if __name__ == "__main__":
    main()

