import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from loguru import logger
from src.hardware import DEVICE
from src.llm import LLM


def get_context(idx, num_words: int, data_dir: str):
    data_dir = Path(data_dir)
    contexts = []
    filename = os.listdir(data_dir)[idx]
    with open(data_dir / filename) as f:
        return " ".join(f.readlines()[0].split(" ")[:num_words])


artefact_dir = Path(__file__).parent / Path("artefacts")
artefact_dir.mkdir(parents=True, exist_ok=True)

llm = LLM("/data/llms/Llama-2-7b-hf")
llm.model.to(DEVICE)


def f(tau: float, logits: torch.Tensor) -> float:
    return torch.sum(torch.softmax(logits, dim=-1) ** (1 / tau))


temp_masses = defaultdict(lambda: [])
n_repetitions = 1000
temps = np.array(list(range(1, 100))) / 100
n_per_tau = 50
for tau in temps:
    logger.info(f"Processing temp == {tau}")
    while len(temp_masses[tau]) < n_per_tau:
        initial_context = get_context(
            random.randint(0, 49999), random.randint(1, 200), "./data/imdb/train/unsup"
        )
        tokens = llm.encode(initial_context).to(DEVICE)
        next_token_logits = llm.get_predicted_next_token_logits(tokens).to(DEVICE)
        temp_mass = float(f(tau, next_token_logits))
        logger.info(f"Temp mass: {temp_mass}")
        temp_masses[tau].append(temp_mass)
        logger.info(f"Generated {len(temp_masses[tau])} of {n_per_tau}")

for tau in temps:
    logger.info(f"tau={tau}, mean mass={mean(temp_masses[tau])}")

with open(artefact_dir / "temp_masses.json", "w") as f:
    f.write(json.dumps(dict(temp_masses), indent=4))
