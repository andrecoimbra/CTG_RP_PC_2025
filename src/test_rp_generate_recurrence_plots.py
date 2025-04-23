#!/usr/bin/env python
# coding: utf-8

import os
import json
from config_local import *
from generate_recurrence_images import (
    gen_recurrence_params,
    generate_rp_images_segment,
)

IMAGES_DIR = "../content/images"
RECORDINGS_DIR = "../content/ctu-uhb-ctgdb"

POLICY = "late_valid"  # 'best_quality', 'early_valid', 'late_valid'
rp_params = gen_recurrence_params(
    dimensions=[2], time_delays=[1], percentages=[6], use_clip_vals=[False]
)

# generate_rp_images(RECORDINGS_DIR, images_dir=IMAGES_DIR, rp_params=rp_params,
#                    policy=POLICY, show_signal=False, show_image=False, verbose=False,
#                    cmap='binary', limit=1, max_seg_min=10)

generate_rp_images_segment(
    RECORDINGS_DIR,
    images_dir=IMAGES_DIR,
    rp_params=rp_params,
    policy=POLICY,
    show_signal=True,
    show_image=True,
    verbose=False,
    cmap="plasma",
    limit=8,
    max_seg_min=10,
    n_dec=2,
)

# generate_rp_images_simple(RECORDINGS_DIR, images_dir=IMAGES_DIR, rp_params=rp_params,
#                           policy=POLICY, show_signal=False, show_image=True, verbose=False,
#                           cmap='binary', limit=10, max_seg_min=10, n_dec=1)
# 1002, 1008
with open(os.path.join(IMAGES_DIR, "rp_images_index.json"), "r") as infile:
    data = json.load(infile)

print("Recordings:", len(data.keys()))
print("Sample Recording :", data["1001"])

total = 0
for k in data.keys():
    for i in data[k]["names"]:
        total += 1
print("Total Images", total)
