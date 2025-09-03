#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import pymap3d as pm
import torch
from . import logger


def get_gps_pos(image_path: Path) -> Tuple[float, float, float]:
    metadata = json.loads(image_path.with_suffix(".json").read_text())
    name = image_path.with_suffix(".json").name
    if "gps_lat" not in metadata:
        gps_path = Path("/vast/ro38seb/datasets/fortepan_gps") / name
        if gps_path.exists():  # manual (llm+geocoding) gps fallback
            metadata = json.loads((gps_path).read_text())

            if "gps_lat" not in metadata:
                print("no gps", gps_path)
            # else:
            #     print("gps", gps_path)
        if "gps_lat" not in metadata:  # if still no gps return 0
            # print(image_path)
            return 0, 0, 0
    lat = float(metadata["gps_lat"][0])
    long = float(metadata["gps_lng"][0])

    # print(lat, long)
    alt = 0

    # if not found:
    #     raise Exception('Did not find metadata for {}'.format(image_path))

    return lat, long, alt


def main(output,
         image_dir: Path,
         image_list: Path,
         closest_geo: int):
    ts = []
    zeros_mask = []
    ref_lat = 47.507311  # parliament, could be None
    ref_long = 19.045654
    ref_alt = 0
    image_list = image_list.read_text().strip().split("\n")
    image_ids = []
    for i, image_id in enumerate(image_list):
        lat, long, alt = get_gps_pos(image_dir / image_id)
        if ref_lat is None:
            ref_lat = lat
            ref_long = long
            ref_alt = alt
        ts.append(torch.FloatTensor(pm.geodetic2ned(lat, long, alt, ref_lat, ref_long, ref_alt)).unsqueeze(0))
        zeros_mask.append(torch.tensor([lat != 0 and long != 0]))
        image_ids.append(i)

    logger.info(f'Obtaining pairwise distances between {len(image_list)} images...')

    ts = torch.cat(ts)
    zero_mask_list = torch.cat(zeros_mask)

    pos_dist = torch.cdist(ts, ts)
    dist_mask = torch.zeros_like(pos_dist)  # true if one of the positions is 0 (invalid)

    dist_mask[~zero_mask_list, :] = True
    dist_mask[:, ~zero_mask_list] = True

    pos_dist = pos_dist + 1_000_000 * dist_mask  # add 100 km to dist of invalid

    pairs = []
    pbar = tqdm(range(len(image_list)))
    total_pairs = 0
    for i in pbar:
        if not zeros_mask[i]:  # skip invalid location
            continue
        dist, closest_pos = torch.topk(pos_dist[i], closest_geo + 1, largest=False)  # closest 1000
        mask = dist < 500  # 1km / 500m radius
        total_pairs += mask.sum()
        pbar.set_postfix({'pairs': mask.sum(), "total": total_pairs})
        # print(ts[i], dist[mask])
        for j in closest_pos[mask]:
            if i != j:
                pairs.append((image_list[image_ids[i]], image_list[image_ids[j]]))

    # for i in pbar:
    #     if not zeros_mask[i]:  # skip invalid location
    #         continue
    #     mask = pos_dist[i] < 500  # 1km radius
    #
    #     total_pairs += mask.sum()
    #     pbar.set_postfix({'pairs': mask.sum(), "total": total_pairs})
    #     for j, match in enumerate(mask):
    #         if match and i != j:
    #             pairs.append((image_list[image_ids[i]], image_list[image_ids[j]]))

    logger.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join(p) for p in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--image_list', required=True, type=Path)
    parser.add_argument('--image_dir', required=True, type=Path)
    parser.add_argument('--closest_geo', required=True, type=int)

    args = parser.parse_args()
    main(**args.__dict__)
