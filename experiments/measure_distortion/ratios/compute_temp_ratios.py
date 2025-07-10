import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from src.distributions import compute_p_distribution, compute_temp_q_distribution
from src.llm import LLM

if __name__ == "__main__":
    data_dir = Path("./data/custom_generated")
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(__file__).parent.parent / Path("results")
    result_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM("/data/llms/Llama-2-7b-hf")
    temp = int(sys.argv[1]) / 100

    initial_contexts = ["The "]
    results = {}
    n_repetitions = 1000

    for initial_context in initial_contexts:
        p_ratios = []
        q_ratios = []
        for _ in range(n_repetitions):
            completions = llm.get_completion(
                initial_context,
                num_return_sequences=2,
                temperature=temp,
                num_new_tokens=100,
            )

            p_dist_1 = compute_p_distribution(
                completion=completions[0],
                llm=llm,
            )
            q_dist_1 = compute_temp_q_distribution(
                completion=completions[0],
                temp=temp,
                llm=llm,
            )
            p_dist_2 = compute_p_distribution(
                completion=completions[1],
                llm=llm,
            )
            q_dist_2 = compute_temp_q_distribution(
                completion=completions[1],
                temp=temp,
                llm=llm,
            )

            logger.info(f"Intiial context: {initial_context}")
            logger.info(f"Completion 1: {completions[0]}")
            logger.info(f"Completion 2: {completions[1]}")

            logger.info(f"p(T_1) (global): {p_dist_1}")
            logger.info(f"p(T_2) (global): {p_dist_2}")
            logger.info(f"q(T_1) (temp-sampling): {q_dist_1}")
            logger.info(f"q(T_2) (temp-sampling): {q_dist_2}")

            p_ratio = np.prod(
                [p_dist_1[idx] / p_dist_2[idx] for idx in range(len(p_dist_1))]
            )
            q_ratio = np.prod(
                [q_dist_1[idx] / q_dist_2[idx] for idx in range(len(q_dist_1))]
            )
            p_ratios.append(float(p_ratio))
            q_ratios.append(float(q_ratio))
            logger.info(f"P ratio: {p_ratio}")
            logger.info(f"Q ratio: {q_ratio}")

        results[initial_context] = {
            "p_ratios": p_ratios,
            "q_ratios": q_ratios,
            "mean_p_ratio": sum(p_ratios) / len(p_ratios),
            "mean_q_ratio": sum(q_ratios) / len(q_ratios),
            "temp": temp,
        }

    with open(result_dir / f"temp={temp}_ratios_long.json", "w") as f:
        f.write(json.dumps(results, indent=4))
