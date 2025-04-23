#!/usr/bin/env python
# coding: utf-8

import os
import json
from config_local import *
from generate_poincare_images import generate_pc_images_segment

IMAGES_DIR = "../content/images"
RECORDINGS_DIR = "../content/ctu-uhb-ctgdb"

POLICY = "late_valid"  # 'best_quality', 'early_valid', 'late_valid'

# generate_pc_images(RECORDINGS_DIR, images_dir=IMAGES_DIR,
#                    policy=POLICY,
#                    show_signal=False, show_image=True, verbose=True, cmap='binary',
#                    limit=3, max_seg_min=10, clip_stage_II=False
#                    )
pc_lags = [1]
# pc_lags = list(range(1, 2))
# pc_lags = list(range(1, 11))


results = generate_pc_images_segment(
    RECORDINGS_DIR,
    images_dir=IMAGES_DIR,
    pc_lags=pc_lags,
    policy=POLICY,
    show_signal=False,
    show_image=True,
    verbose=False,
    # cmap="jet",xx
    limit=1,
    max_seg_min=40,
    n_dec=4,
)

print(results)

# generate_pc_images_simple(RECORDINGS_DIR, images_dir=IMAGES_DIR,
#                           policy=POLICY, show_signal=False, show_image=True, verbose=False,
#                           cmap='binary', limit=5, max_seg_min=10, n_dec=1, clip_stage_II=False)

with open(os.path.join(IMAGES_DIR, "pc_images_index.json"), "r") as infile:
    data = json.load(infile)

# print('Recordings:', len(data.keys()))
# print('Sample Recording :', data['1001'])

# print(data.keys())
# tally = 0
# for k in data.keys():
#     print(data[k])
# print('Total Images', tally)
