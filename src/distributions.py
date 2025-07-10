from typing import List, Dict, Any

import numpy as np

from src.llm import LLM
from src.metrics import compute_metrics
from src.hardware import DEVICE
from loguru import logger
import torch

def compute_p_distribution(completion: Dict[str, Any], llm: LLM) -> List[float]:
    n_lls = compute_metrics(completion["full_tokens"], llm)["negative_log_likelihoods"]
    len_context = completion["context_tokens"].shape[1]
    n_lls_completions = n_lls[len_context - 1 :]
    likelihoods = [np.exp(-x).item() for x in n_lls_completions]
    return likelihoods


def compute_top_k_next_token_probs(
    logits: torch.Tensor, next_token_idx: int, llm: LLM, top_k: int
) -> float:
    top_k_sample = llm.sample_top_k_from_logits(logits, top_k=top_k)
    for idx in range(len(top_k_sample["token_indices"])):
        if top_k_sample["token_indices"][idx] == next_token_idx.item():
            return top_k_sample["token_probs"][idx] / sum(top_k_sample["token_probs"])
    return None


def compute_top_k_completion_probability(
    completion: Dict[str, Any], llm: LLM, top_k: int
):
    context_tokens = completion["context_tokens"]
    completion_tokens = completion["completion_tokens"]
    full_tokens = completion["full_tokens"]
    output = []
    completion_size = completion_tokens.shape[1]
    with torch.no_grad():
        outputs = llm.model(full_tokens.to(DEVICE))
    for idx in range(completion_size):
        target_token = full_tokens[..., idx + context_tokens.shape[1]]
        next_prob = compute_top_k_next_token_probs(
            logits=outputs["logits"][0, idx + context_tokens.shape[1] - 1, :],
            next_token_idx=target_token,
            llm=llm,
            top_k=top_k,
        )
        output.append(next_prob)
    return output


def compute_top_k_q_distribution(
    completion: Dict[str, Any], llm: LLM, top_k: int
) -> List[float]:
    return compute_top_k_completion_probability(
        completion=completion, llm=llm, top_k=top_k
    )


def compute_top_p_next_token_probs(
    logits: torch.Tensor, next_token_idx: int, llm: LLM, top_p: int
) -> float:
    top_p_sample = llm.sample_top_p_from_logits(logits, top_p=top_p)
    for idx in range(len(top_p_sample["token_indices"])):
        if top_p_sample["token_indices"][idx] == next_token_idx.item():
            return top_p_sample["token_probs"][idx] / sum(top_p_sample["token_probs"])
    return None


def compute_top_p_completion_probability(completion: str, llm: LLM, top_p: int):
    context_tokens = completion["context_tokens"]
    full_tokens = completion["full_tokens"]
    output = []
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
        output.append(next_prob)
    return output


def compute_top_p_q_distribution(completion: str, llm: LLM, top_p: int) -> List[float]:
    return compute_top_p_completion_probability(
        completion=completion, llm=llm, top_p=top_p
    )

def compute_temp_next_token_prob(
    logits: torch.Tensor,
    next_token_idx: int,
    temp: float,
) -> float:
    all_token_scaled_probs = torch.softmax(logits / temp, dim=-1)
    return all_token_scaled_probs[next_token_idx.item()]


def compute_temp_completion_probability(completion: str, llm: LLM, temp: float):
    context_tokens = completion["context_tokens"]
    full_tokens = completion["full_tokens"]
    output = []
    completion_size = completion["completion_tokens"].shape[1]
    with torch.no_grad():
        outputs = llm.model(full_tokens.to(DEVICE))
    for idx in range(completion_size):
        sequence_to_process = full_tokens[..., : idx + context_tokens.shape[1]]
        target_token = full_tokens[..., idx + context_tokens.shape[1]]
        logger.info(f"Processing: {sequence_to_process}")
        logger.info(f"Target token {target_token}")
        next_prob = compute_temp_next_token_prob(
            logits=outputs["logits"][0, idx + context_tokens.shape[1] - 1, :],
            next_token_idx=target_token,
            temp=temp,
        )
        output.append(float(next_prob.item()))
    logger.info(f"Output: {output}")
    return output


def compute_temp_q_distribution(completion: str, llm: LLM, temp: float) -> List[float]:
    return compute_temp_completion_probability(
        completion=completion, llm=llm, temp=temp
    )
