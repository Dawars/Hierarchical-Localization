#!/usr/bin/env python
# SBATCH -N 1 # n nodes
# SBATCH --ntasks-per-node=8 # cpu cores
# SBATCH --mem=500GB
# SBATCH --partition=gpu
# SBATCH --gres=gpu:1
# SBATCH -x gpu005
import re
import json
from tqdm import tqdm
import itertools
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def naive_json_from_text(text):
    # Regular expression to match JSON objects
    json_pattern = r'\{[\s\S]*}'
    # json_pattern = r'\{(?:[^{}]|(?R))*\}'

    # Find all matches of JSON objects in the text
    json_matches = re.findall(json_pattern, text)

    # Attempt to parse each matched JSON object
    valid_json_objects = []
    for match in json_matches:
        try:
            valid_json = json.loads(match)
            valid_json_objects.append(valid_json)
        except json.JSONDecodeError:
            pass  # Skip invalid JSON objects
    if not valid_json_objects:
        print(json_matches)
    return valid_json_objects

    # match = re.search(r'\{[\s\S]*}', text)
    # if not match:
    #     print("no match")
    #     return None
    #
    # try:
    #     return json.loads(match.group(0))
    # except json.JSONDecodeError as e:
    #     print("Error: ", e)
    #     print(match)
    #     return None


descriptions = [
    # "Kálvin tér, a Fővárosi Moziüzemi Vállalat (FŐMO) által forgalmazott film plakátja, jobbra a Kecskeméti utca torkolata. A tűzfalon az egykori városkapu emlékét idézi a téglából kirakott ábrázolás.",
    # "a József körút a Rákóczi térnél. A villamoson a Fővárosi Moziüzemi Vállalat (FŐMO) által forgalmazott film plakátja.",
    # "Múzeum körút az Astoria kereszteződés közelében, a palánk mögött a metróépítés területe, jobbra a háttérben az ELTE épülete. A Fővárosi Moziüzemi Vállalat (FŐMO) által forgalmazott film plakátja.",
    # "Múzeum körút az Astoria kereszteződés közelében, a palánk mögött a metróépítés területe. A Fővárosi Moziüzemi Vállalat (FŐMO) által forgalmazott film plakátja.",
    # "Thököly út, a Fővárosi Moziüzemi Vállalat (FŐMO) által forgalmazott filmek hirdetőoszlopa a Murányi utcai kereszteződés előtt.",
    # "Thököly út, a Fővárosi Moziüzemi Vállalat (FŐMO) által forgalmazott filmek hirdetőoszlopa a 24-es számú ház előtt.",
]
image_ids = []

for json_path in Path("/vast/ro38seb/datasets/fortepan").glob("*.json"):
    metadata = json.loads(json_path.read_text())
    if "gps_lat" not in metadata and "description" in metadata:
        descriptions.append(metadata["description"][0].replace(r"\"", "").replace("\"", ""))
        image_ids.append(json_path.with_suffix("").name)
print(descriptions)

Path("/work/ro38seb/datasets/fortepan/descriptions.txt").write_text("\n".join(descriptions))

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

messages = [
    {"role": "user", "content": """You are a text parsing assistant. You take an image text description written in Hungarian and
     extract the address(es) from the given text and output it. If there is no address return nothing.  
    Your output must be in valid JSON. Do not output anything other than the JSON.
So for instance the following: Pestújhelyi út, szemben a Sztárai Mihály (Cservenka Miklós) téri Szolgáltatóház. Jobbra az épület mögött a Neptun utca.
would be converted to:"""},
    {"role": "assistant", "content":
        '''
        { 
            "address": ["Pestújhelyi út", "Sztárai Mihály (Cservenka Miklós) tér", "Neptun utca"]
        }
        '''
     },
    {"role": "user", "content": """So for instance the following: Endrődi Sándor utca 18/a, Szilágyi villa. 
    would be converted to:"""},
    {"role": "assistant", "content":
        '''
        { 
            "address": ['Endrődi Sándor utca 18/a', 'Szilágyi villa']
        }
        '''
     },
    {"role": "user", "content": """So for instance the following: az Utazás a koponyám körül című film főszereplője, az egyik lázálom jelenetében Latinovits Zoltán színművész. 
    would be converted to the following since there is no address provided:"""},
    {"role": "assistant", "content":
        '''
        { 
            "address": []
        }
        '''
     },
    {"role": "user", "content": """So for instance the following: 'Szent István körút, jobbra a Hegedűs Gyula utca. A kép forrását kérjük így adja meg: Fortepan / Budapest Főváros Levéltára. Levéltári jelzet: HU.BFL.XV.19.c.10 
    would be converted to the following since there is no address provided:"""},
    {"role": "assistant", "content":
        '''
        { 
            "address": ["Szent István körút", "Hegedűs Gyula utca"]
        }
        '''
     },
    {"role": "user", "content": """So for instance the following: Orgona út 1., Páduai Szent Antal-kápolna. Szemben a szentélyfalon a „Betlehem, három királyok” szárnyas oltár, Udvardi Erzsébet 1976-ban felszentelt alkotása. 
    would be converted to the following since there is no address provided:"""},
    {"role": "assistant", "content":
        '''
        { 
            "address": ["Orgona út 1."]
        }
        '''
     },
    {"role": "user", "content": """So for instance the following: Vadvíz utca - Forster János Jakab utca sarok, a Laji Csárda a Ráckevei (Soroksári)-Duna partján. 
    would be converted to the following since there is no address provided:"""},
    {"role": "assistant", "content":
        '''
        { 
            "address": ["Vadvíz utca", "Forster János Jakab utca sarok", "Laji Csárda"]
        }
        '''
     },
    {"role": "user", "content": """So for instance the following: a Slussen - Djurgården hajójárat Nybrohamnen-ről fotózva, háttérben balra a Strandvägen 7. és 9-es számú ház, jobbra a Nordiska Museet. 
    would be converted to the following since there is no address provided:"""},
    {"role": "assistant", "content":
        '''
        { 
            "address": ["Nybrohamnen", "Strandvägen 7.", "Strandvägen 9.", "Nordiska Museet"]
        }
        '''
     },
]

data = {}
pbar = tqdm(zip(image_ids, descriptions))
with Path("/work/ro38seb/datasets/fortepan/addresses_2.txt").open("w") as f:
    for name, desc in pbar:
        if not desc or desc == "":
            continue
        pbar.set_postfix({"name": name, "desc": desc})
        encodeds = tokenizer.apply_chat_template(messages + [{"role": "user", "content": str(desc)}],
                                                 return_tensors="pt")
        # print(encodeds)
        model_inputs = encodeds.to(device)
        model.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=100)
        # print(generated_ids)
        decoded = tokenizer.batch_decode(generated_ids[:, model_inputs.shape[1]:])
        # print("-------------decoded text-------------")

        addresses = naive_json_from_text(decoded[0])
        if not addresses:
            # print(decoded[0])
            # print("---------------------------------")
            continue

        print("Address", addresses)

        data[name] = addresses
        # print("---------------------------------")
        if addresses[0].get("address", "") not in ["", None, [], [""]]:
            # print("writing")
            f.write(json.dumps({"name": name, "address": addresses[0]["address"]}, ensure_ascii=False) + "\n")
            f.flush()

Path("/work/ro38seb/datasets/fortepan/addresses.txt").write_text(json.dumps(data, indent=4, ensure_ascii=False))
