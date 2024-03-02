import json
from collections import defaultdict
from pathlib import Path

data = json.loads(Path("/Users/dawars/datasets/addresses_accent.txt").read_text())

address_set = set()
name_to_addresses = defaultdict(list)
address_to_name = defaultdict(list)

for name, address in data.items():
    if "address" not in address[0]:
        continue
    address_list = address[0]["address"]

    metadata = json.loads(Path("/Users/dawars/datasets/fortepan/" + name + ".json").read_text())
    country = metadata.get("orszag_name", [None])[0]
    city = metadata.get("varos_name", [None])[0]
    helyszin = metadata.get("helyszin_name", [None])[0]
    if helyszin:  # use instead of addr
        complete_addr = (helyszin.lower(), city or "", country or "")
        key = '*'.join(complete_addr)

        address_set.add(key)
        address_to_name[key].append(name[len("fortepan_"):])  # reverse index
        name_to_addresses[name[len("fortepan_"):]].append(key)  # reverse index
        continue

    local_addr = []
    for addr in address_list:
        to_remove = ["A kép forrását kérjük így adja meg:",
                     "Fortepan / Budapest Főváros Levéltára.", "Levéltári jelzet:", "HU.BFL.XV.19.c.10", "HU.BFL."]

        for word in to_remove:
            if word in addr:
                continue

        clean = ''.join(e for e in addr if e.isalnum() or e == " ").strip().lower()

        try:
            int(clean)  # only house number
            if local_addr:
                local_addr[-1] += " " + clean

            continue
        except:
            pass

        if clean == "" or clean == []:
            continue

        local_addr.append(clean)
    for x in local_addr:
        complete_addr = (x, city or "", country or "")
        key = '*'.join(complete_addr)
        address_set.add(key)  # save as tuple
        address_to_name[key].append(name[len("fortepan_"):])  # reverse index
        name_to_addresses[name[len("fortepan_"):]].append(key)  # reverse index

print(len(address_set))
# print(address_set)
Path("/Users/dawars/datasets/fortepan_to_addresses.txt").write_text(json.dumps(name_to_addresses, ensure_ascii=False))
Path("/Users/dawars/datasets/addresses_to_fortepan_id.txt").write_text(json.dumps(address_to_name, ensure_ascii=False))
Path("/Users/dawars/datasets/addresses_set.txt").write_text(json.dumps(list(address_set), ensure_ascii=False))
