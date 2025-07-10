import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from loguru import logger
from src.llm import LLM


def get_context(idx, num_words: int, data_dir: str):
    data_dir = Path(data_dir)
    contexts = []
    filename = os.listdir(data_dir)[idx]
    with open(data_dir / filename) as f:
        return " ".join(f.readlines()[0].split(" ")[:num_words])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

artefact_dir = Path(__file__).parent / Path("artefacts")
artefact_dir.mkdir(parents=True, exist_ok=True)

llm = LLM("/data/llms/Llama-2-7b-hf")
llm.model.to(device)

top_k_masses = defaultdict(lambda: [])
n_repetitions = 1000
ks = list(range(1, 155, 1))
n_per_k = 50
for k in ks:
    logger.info(f"Processing k == {k}")
    while len(top_k_masses[k]) < n_per_k:
        initial_context = get_context(
            random.randint(0, 49999), random.randint(1, 200), "./data/imdb/train/unsup"
        )
        tokens = llm.encode(initial_context).to(device)
        next_token_logits = llm.get_predicted_next_token_logits(tokens)
        top_k_logits = llm.sample_top_k_from_logits(next_token_logits, top_k=k)
        top_k_mass = sum(top_k_logits["token_probs"])
        top_k_masses[k].append(top_k_mass)
        logger.info(f"Generated {len(top_k_masses[k])} of {n_per_k}")

for k in ks:
    logger.info(f"k={k}, mean mass={mean(top_k_masses[k])}")

with open(artefact_dir / "top_k_masses.json", "w") as f:
    f.write(json.dumps(dict(top_k_masses), indent=4))
