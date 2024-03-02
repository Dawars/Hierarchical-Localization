#!/usr/bin/env python
#SBATCH -N 1 # n nodes
#SBATCH --ntasks-per-node=8 # cpu cores
#SBATCH --job-name=geocoding
#SBATCH --mem=100GB
import json
import urllib
from time import sleep

import requests
from urllib.parse import urlencode, quote_plus

from pathlib import Path

from tqdm import tqdm

api_key = ""

addresses = json.loads(Path("/home/ro38seb/datasets/fortepan/addresses_set.txt").read_text())

gps_coords = {}
with Path("/home/ro38seb/datasets/fortepan/gps_coords_lines.txt").open("w") as f:
    for key in tqdm(addresses):
        address, city, country = key.split("*")

        query = address
        country_query = ""
        if city != "":
            if "Budapest" in city and city[-1] == ".":
                city = city.replace("Budapest ", "") + " kerület, Budapest"
            query += ", " + city
        if country != "":
            query += ", " + country
            country_query = f"components=country:{country}&"
        res = requests.get(
            f"https://maps.googleapis.com/maps/api/geocode/json?address={quote_plus(query)}&{country_query}language=hu&key={api_key}")
        if res.ok:
            data = res.json()["results"]
            # print(data)

            gps_coords[key] = data

            f.write(json.dumps({"address": key, "gps": data}, ensure_ascii=False) + "\n")
            f.flush()
        else:
            print(res.status_code, key)
        sleep(5)

Path("/home/ro38seb/datasets/fortepan/gps_coords.txt").write_text(json.dumps(gps_coords, ensure_ascii=False))
Path("/home/ro38seb/datasets/fortepan/gps_coords_readable.txt").write_text(json.dumps(gps_coords, indent=4, ensure_ascii=False))

print("done")
