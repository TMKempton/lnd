import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from loguru import logger
from src.distributions import compute_p_distribution, compute_temp_q_distribution
from src.llm import LLM

if __name__ == "__main__":
    data_dir = Path("./data/custom_generated")
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(__file__).parent / Path("results")
    result_dir.mkdir(parents=True, exist_ok=True)
    artefact_dir = Path(__file__).parent / Path("artefacts")
    artefact_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM("/data/llms/Llama-2-7b-hf")
    initial_contexts = ["The "]
    results = {}
    temp = int(sys.argv[1]) / 100
    NEW_TOKENS_PER_COMPLETION=int(sys.argv[2])
    NUM_COMPLETIONS = int(sys.argv[3])

    logger.info(f"Processing temp={temp}")
    for initial_context in initial_contexts:
        logger.info(f"Processing context `{initial_context}`")
        entropies = []
        perplexities = []
        completions = []
        for _ in range(NUM_COMPLETIONS // 10):
            completions.extend(
                llm.get_completion(
                    initial_context,
                    num_return_sequences=10,
                    temperature=temp,
                    num_new_tokens=NEW_TOKENS_PER_COMPLETION,
                )
            )

        with open(artefact_dir / f"ln_temp={temp}_completion.json", "w") as f:
            f.write(
                json.dumps(
                    {i: (completions[i]["context"] + completions[i]["generated_text"])  for i in range(len(completions))}, indent=4
                )
            )

        for completion in completions:
            p_dist = compute_p_distribution(
                completion=completion, llm=llm
            )
            q_dist = compute_temp_q_distribution(
                completion=completion,
                temp=temp,
                llm=llm,
            )
            n = len(p_dist)
            perplexity = -(1 / n) * np.sum(np.log(np.array(p_dist)))
            entropy = -(1 / n) * np.sum(np.log(np.array(q_dist)))
            perplexities.append(float(perplexity))
            entropies.append(float(entropy))

            logger.info(f"Perplexity {perplexity}")
            logger.info(f"Entropy: {entropy}")

        results[initial_context] = {
            "entropies": entropies,
            "perplexity": perplexities,
            "mean_entropy": mean(entropies),
            "mean_perplexity": mean(perplexities),
            "temp": temp,
        }

    with open(result_dir / f"ln_temp={temp}_metrics.json", "w") as f:
        f.write(json.dumps(results, indent=4))
