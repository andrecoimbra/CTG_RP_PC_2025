#!/usr/bin/env python
# coding: utf-8

import os
import json
from pprint import pprint

import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt

import wfdb
from ctg_utils import get_all_recno, parse_meta_comments
from basic_denoise import (
    get_valid_segments,
    get_segment_concatenation,
    get_segment_removing_zeros,
)
from libPC import create_pc

POLICIES = ["best_quality", "early_valid", "late_valid"]

def generate_pc_images_segment(
    recordings_dir,
    n_dec=4,
    clip_stage_II=False,
    max_seg_min=10,
    policy="early_valid",
    pc_lags=[1],
    images_dir="",
    images_index_file="pc_images_index.json",
    show_signal=False,
    show_image=False,
    verbose=False,
    cmap=None,
    limit=-1,
):

    assert policy in POLICIES

    if images_dir and not os.path.exists(images_dir):
        os.mkdir(images_dir)

    results = {}
    for recno in sorted(get_all_recno(recordings_dir)):
        if limit == 0:
            break
        limit -= 1

        # if recno != '1005':
        #     # print(recno)
        #     continue

        recno_full = os.path.join(recordings_dir, recno)
        all_sig, meta = wfdb.io.rdsamp(recno_full)
        meta = parse_meta_comments(meta["comments"])
        if verbose:
            print(
                "\nRecord: {}  Samples: {}   Duration: {:0.1f} min   Stage.II: {} min".format(
                    recno,
                    all_sig.shape[0],
                    all_sig.shape[0] / 4 / 60,
                    meta["Delivery"]["II.stage"],
                )
            )

        sig_hr = all_sig[:, 0]
        if clip_stage_II and meta["Delivery"]["II.stage"] != -1:
            idx = int(meta["Delivery"]["II.stage"] * 60 * 4)
            sig_hr = sig_hr[:-idx]
        ts = np.arange(len(sig_hr)) / 4.0

        # select concatenation of segments as close to the chosen length
        segment = get_segment_concatenation(
            sig_hr, ts, recno, max_seg_min=max_seg_min, policy=policy
        )

        if len(segment) == 0:
            continue

        if show_signal:
            seg_start = segment["seg_start"]
            seg_end = segment["seg_end"]
            seg_hr = segment["seg_hr"]
            seg_tm = segment["seg_ts"] / 60
            orig_seg_hr = segment["orig_seg_hr"]

            seg_tm = np.arange(len(seg_hr)) / 4.0 / 60

            plt.figure(figsize=(12, 3.8))
            plt.title(
                "{}: Final Segment  {}-{} (considered length = {:0.1f} min)".format(
                    recno, seg_start, seg_end, len(seg_hr) / 60 / 4
                )
            )
            plt.plot(seg_tm, seg_hr)
            plt.plot(seg_tm, orig_seg_hr, alpha=0.25)
            plt.xlim(seg_tm[0], seg_tm[-1])
            plt.ylim(50, 200)
            plt.xlabel("Time [seconds]")
            plt.ylabel("FHR [bpm]")
            plt.show()

            print("Segment length: {:0.1f} min".format(len(seg_hr) / 60 / 4))

        selected_hr = segment["seg_hr"]

        # Reduce signal sampling rate (compact)
        if n_dec > 1:
            selected_hr = scipy.signal.decimate(selected_hr, n_dec)
            # selected_hr = selected_hr[::n_dec]
            # selected_hr = scipy.signal.resample(selected_hr, len(selected_hr) // n_dec)

        selected_hr = np.round(selected_hr).astype(int)

        if (len(selected_hr) * n_dec / 4 / 60) < max_seg_min:
            print(
                f"Segment {recno} only last {(len(selected_hr) * n_dec / 4 / 60):.2f} min, therefore less than desired duration ({max_seg_min})."
            )

        image_names = []
        for lag in pc_lags:
            fname = create_pc(
                selected_hr,
                base_name=recno,
                show_image=show_image,
                images_dir=images_dir,
                cmap=cmap,
                use_clip=clip_stage_II,
                lag=lag,
            )
            image_names.append(fname)

        results[recno] = {"names": image_names, "outcome": meta["Outcome"]}

    if verbose:
        pprint(results)

    with open(os.path.join(images_dir, images_index_file), "w") as outfile:
        json.dump(results, outfile)

    num_records = len(results)
    num_images = num_records * len(pc_lags)

    return num_records, num_images