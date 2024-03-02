import json
from collections import defaultdict
from pathlib import Path

gps_coords = json.loads(Path("/Users/dawars/datasets/gps_coords.txt").read_text())
addresses = json.loads(Path("/Users/dawars/datasets/fortepan_to_addresses.txt").read_text())
gps_output = Path("/Users/dawars/datasets/fortepan_gps")
# todo process gps for individual addresses
address_to_gps = {}
for address, gps_data in gps_coords.items():
    place, city, country = address.split("*")

    # print(len(gps_data))
    if not gps_data:
        print(address, "missing")
        continue
    gps = gps_data[0]["geometry"]["location"]
    address_to_gps[address] = gps
    # print("partial_match" in gps_data[0], gps_data[0]["types"])

i = 0
# todo use first available for now for testing
for name, address_data in addresses.items():
    addr = address_data[0]
    if addr not in address_to_gps:
        continue
    gps = address_to_gps[addr]
    print(gps)
    i += 1

    gps_json = {
        "gps_lat": [
            gps["lat"]
        ],
        "gps_lng": [
            gps["lng"]
        ],
    }

    (gps_output / f"fortepan_{name}.json").write_text(json.dumps(gps_json))

print(i)
