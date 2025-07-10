import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import List

import numpy as np
from loguru import logger

from src.distributions import compute_p_distribution
from src.llm import LLM
from src.hardware import DEVICE


def is_valid_temp_completion(
    completion: str, llm: LLM, temp: int
) -> bool:
    p_dist = compute_p_distribution(
        completion=completion, llm=llm
    )
    prod = np.prod(p_dist) ** ((1 / temp) - 1)
    logger.info(f"Prob={prod}")
    random_num = random.uniform(0, 1)
    if random_num < prod:
        return True
    else:
        return False


def filter_completions(
    completions: List[str], llm: LLM, temp: int
) -> List[str]:
    return [c for c in completions if is_valid_temp_completion(c, llm, temp)]


if __name__ == "__main__":
    data_dir = Path("./data/custom_generated")
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(__file__).parent / Path("results")
    result_dir.mkdir(parents=True, exist_ok=True)
    artefact_dir = Path(__file__).parent / Path("artefacts")
    artefact_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM("/data/llms/Llama-2-7b-hf")
    llm.model.to(DEVICE)

    initial_contexts = ["The "]
    results = {}
    temp = int(sys.argv[1]) / 100
    NEW_TOKENS_PER_COMPLETION=int(sys.argv[2])
    NUM_COMPLETIONS = int(sys.argv[3])
    logger.info(f"Processing temp={temp}")
    logger.info(f"New tokens: {NEW_TOKENS_PER_COMPLETION}")
    logger.info(f"Num completions: {NUM_COMPLETIONS}")

    for initial_context in initial_contexts:

        logger.info(f"Processing context `{initial_context}`")
        entropies = []
        perplexities = []
        completions = []
        num_generated = 0
        num_rejected = 0
        gens_per_step = 10
        context_tokens = llm.encode(initial_context)
        while len(completions) < NUM_COMPLETIONS:
            logger.info(f"Size of completions: {len(completions)} of {NUM_COMPLETIONS}")
            new_completions = llm.fast_pure_sampling(context=initial_context, num_new_tokens=NEW_TOKENS_PER_COMPLETION)
            num_generated += gens_per_step
            logger.info("Filtering completion batch")
            filtered_completions = filter_completions(
                completions=new_completions,
                llm=llm,
                temp=temp,
            )
            num_rejected += len(new_completions) - len(filtered_completions)
            completions.extend(filtered_completions)

        completions = completions[:NUM_COMPLETIONS]
        rejection_rate = num_rejected / num_generated
        acceptance_rate = 1 - rejection_rate
        logger.info(f"Rejection rate = {rejection_rate}")

        with open(artefact_dir / f"gn_temp={temp}_completion.json", "w") as f:
            f.write(
                json.dumps(
                    {i: (completions[i]["context"] + completions[i]["generated_text"]) for i in range(len(completions))}, indent=4
                )
            )

        for completion in completions:
            p_dist = compute_p_distribution(
                completion=completion, llm=llm
            )
            logger.info(p_dist)
            n = len(p_dist)
            perplexity = -(1 / n) * np.sum(np.log(np.array(p_dist)))
            perplexities.append(float(perplexity))
            entropy = (1 / temp) * perplexity + (1 / n) * np.log(acceptance_rate)
            entropies.append(float(entropy))

            logger.info(f"Perplexity {perplexity}")
            logger.info(f"Entropy: {entropy}")

        results[initial_context] = {
            "entropies": entropies,
            "perplexity": perplexities,
            "mean_entropy": mean(entropies),
            "mean_perplexity": mean(perplexities),
            "temp": temp,
            "rejection_rate": rejection_rate,
            "acceptance_rate": acceptance_rate,
        }

    with open(result_dir / f"gn_temp={temp}_metrics.json", "w") as f:
        f.write(json.dumps(results, indent=4))
