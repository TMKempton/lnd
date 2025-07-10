import json
import sys
from pathlib import Path
from statistics import mean
from typing import List

import numpy as np
import torch
from loguru import logger

from src.distributions import compute_p_distribution, compute_top_p_next_token_probs
from src.llm import LLM
from src.hardware import DEVICE



def is_valid_top_p_completion(
    completion: str, llm: LLM, top_p: int
) -> bool:
    context_tokens = completion["context_tokens"]
    full_tokens = completion["full_tokens"]
    completion_size = completion["completion_tokens"].shape[1]
    with torch.no_grad():
        outputs = llm.model(full_tokens.to(DEVICE))
    for idx in range(completion_size):
        target_token = full_tokens[..., idx + context_tokens.shape[1]]
        next_prob = compute_top_p_next_token_probs(
            logits=outputs["logits"][0, idx + context_tokens.shape[1] - 1, :],
            next_token_idx=target_token,
            llm=llm,
            top_p=top_p,
        )
        if next_prob is None:
            logger.info(f"Rejecting on {idx} of {completion_size}")
            return False
    return True


def filter_completions(
    completions: List[str], llm: LLM, top_p: int
) -> List[str]:
    return [c for c in completions if is_valid_top_p_completion(c, llm, top_p)]


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
    top_p = int(sys.argv[1]) / 100
    NEW_TOKENS_PER_COMPLETION=int(sys.argv[2])
    NUM_COMPLETIONS = int(sys.argv[3])
    logger.info(f"Processing p={top_p}")
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
                top_p=top_p,
            )
            num_rejected += len(new_completions) - len(filtered_completions)
            completions.extend(filtered_completions)

        completions = completions[:NUM_COMPLETIONS]
        rejection_rate = num_rejected / num_generated
        acceptance_rate = 1 - rejection_rate
        logger.info(f"Rejection rate = {rejection_rate}")

        with open(artefact_dir / f"gn_top_p={top_p}_completion.json", "w") as f:
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
            entropy = perplexity + (1 / n) * np.log(acceptance_rate)
            entropies.append(float(entropy))

            logger.info(f"Perplexity {perplexity}")
            logger.info(f"Entropy: {entropy}")

        results[initial_context] = {
            "entropies": entropies,
            "perplexity": perplexities,
            "mean_entropy": mean(entropies),
            "mean_perplexity": mean(perplexities),
            "top_p": top_p,
            "rejection_rate": rejection_rate,
            "acceptance_rate": acceptance_rate,
        }

    with open(result_dir / f"gn_top_p={top_p}_metrics.json", "w") as f:
        f.write(json.dumps(results, indent=4))
