#!/usr/bin/env python
import argparse
import json
import math
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import pymap3d as pm
import scipy.spatial
import torch
from PIL import Image

from dateutil import parser

from utils.read_write_model import read_images_binary


def get_metadata(image_path: Path) -> Tuple[float, float, float]:
    metadata = json.loads(image_path.with_suffix(".json").read_text())
    name = image_path.with_suffix(".json").name
    # if "gps_lat" not in metadata:
    #     gps_path = Path("/vast/ro38seb/datasets/fortepan_gps") / name
    #     if gps_path.exists():  # manual (llm+geocoding) gps fallback
    #         metadata = json.loads((gps_path).read_text())

    return metadata


def images_from_dir(image_dir: Path):
    return list([path.name for path in image_dir.glob("*.jpg")])


def images_from_list(image_list_path: Path):
    return image_list_path.read_text().strip().split("\n")


def images_from_model(model_path: Path):
    images = read_images_binary(model_path / "images.bin")
    return list([img.name for img in images.values()])


if __name__ == '__main__':
    blacklist = []
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-dinov2-50_long/models/83_group_photo")))
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-dinov2-50_long/models/81_grouphoto")))
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-dinov2-50_long/models/75_tv_interior")))
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-dinov2-50_long/models/69_bus")))
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-dinov2-50_long/models/42_trams_moricz")))

    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/budapest/sfm_mast3r_disk_bp_cluster_0/models/0")))
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/budapest/sfm_mast3r_disk_bp_cluster_1_1000/models/6")))
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/budapest/sfm_mast3r_disk_bp_cluster_2/models/10")))
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/budapest/sfm_mast3r_disk_bp_cluster_7/models/27")))
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/budapest/sfm_mast3r_disk_bp_cluster_14/models/1")))
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/budapest/sfm_mast3r_disk_bp_cluster_15/models/4")))
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/budapest/sfm_mast3r_disk_bp_cluster_19/models/23")))
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/budapest/sfm_mast3r_disk_bp_cluster_14_1000/snapshots/1169590297")))
    blacklist.extend(images_from_model(Path("/Users/dawars/3d_recon/budapest/sfm_mast3r_disk_bp_cluster_24/models/9")))

    print(len(set(blacklist)))
    Path("/Users/dawars/datasets/image_blacklist_budapest.txt").write_text("\n".join(blacklist))

    # budapest_images = []
    # all_images = []
    # for path in Path("/Users/dawars/datasets/fortepan").glob("*.json"):
    #     metadata = get_metadata(path)
    #     name = path.with_suffix(".jpg").name
    #     if "orszag_name" in metadata and metadata["orszag_name"][0] == "Magyarország":
    #         if "varos_name" in metadata and "budapest" in metadata["varos_name"][0].lower():
    #             budapest_images.append(name)
    #     all_images.append(name)
    #
    # print(len(budapest_images))
    #
    # Path("/Users/dawars/datasets/image_list_budapest.txt").write_text("\n".join(budapest_images))
    # blacklist = ["fortepan_82617.jpg", "fortepan_215659.jpg"]  # dups
    # blacklist.extend(images_from_list(Path("blacklist.txt")))  # people images
    # # doppelgangers
    # blacklist.extend(
    #     images_from_dir(Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/-1485266960/pest")))
    # blacklist.extend(images_from_list(Path("/Users/dawars/3d_recon/hloc/blacklist_castle_fish.txt")))  # fishermans
    # blacklist.extend(images_from_list(Path("/Users/dawars/datasets/image_list_theater.txt")))  # kuria + theater
    # blacklist.extend(list(
    #     set(images_from_dir(
    #         Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/kuria_theater/images")))
    #     - set(images_from_dir(
    #         Path("/Users/dawars/3d_recon/hloc/sfm_disk-lightglue_pairs-cosplace-50/kuria_theater/kuria")))
    # ))
    #
    #
    # images = set(budapest_images) - set(blacklist)
    # Path("/Users/dawars/datasets/image_list_budapest_no_pest_fish_theater.txt").write_text("\n".join(images))
    #
    # # non_budapest = set(all_images) - set(budapest_images)
    # # Path("/Users/dawars/datasets/image_list_non_budapest.txt").write_text("\n".join(non_budapest))
