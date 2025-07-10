import json
from statistics import mean

with open("experiments/measure_distortion/masses/artefacts/temp_masses.json", "r") as f:
    data = json.load(f)

for key in list(data.keys())[50:]:
    print(key, mean(data[key]))


with open(
    "experiments/measure_distortion/masses/artefacts/top_k_masses.json", "r"
) as f:
    data = json.load(f)

for key in [list(data.keys())[i] for i in [4, 49, 149]]:
    print(key, mean(data[key]))
