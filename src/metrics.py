from typing import List, Tuple

import torch
from torch.nn import CrossEntropyLoss

from src.hardware import DEVICE
from src.llm import LLM


def compute_metrics(
    tokens: torch.Tensor,
    llm: LLM,
) -> Tuple[float, List[float]]:
    with torch.no_grad():
        logits = llm.model(tokens.to(DEVICE)).logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = tokens[..., 1:].contiguous().to(DEVICE)
    loss_fct = CrossEntropyLoss(reduction="none")
    neg_lls = loss_fct(shift_logits.transpose(1, 2), shift_labels)[0]
    avg_neg_ll = torch.mean(neg_lls)
    perplexity = torch.exp(avg_neg_ll)
    return {
        "perplexity": perplexity.item(),
        "negative_log_likelihoods": neg_lls.tolist(),
    }
