import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plot_dir = Path(__file__).parent / Path("plots")
plot_dir.mkdir(exist_ok=True, parents=True)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True, sharex=True)
for idx, temp in enumerate([0.86, 0.95, 0.98]):
    with open(
        Path(__file__).parent / "results" / f"temp={temp}_ratios_long.json", "r"
    ) as f:
        data = json.load(f)

    q_p_ratio_ratios_temp = []
    for context in data:
        completion_ratios = [
            data[context]["q_ratios"][idx]
            / (data[context]["p_ratios"][idx] ** (1 / temp))
            for idx in range(len(data[context]["p_ratios"]))
        ]
        q_p_ratio_ratios_temp.extend(completion_ratios)
    sns.ecdfplot(
        x=np.abs(np.log(q_p_ratio_ratios_temp)),
        ax=ax[idx],
        label=f"Temp: {temp}",
        linewidth=5,
        color=sns.color_palette()[2],
        alpha=0.7,
    )
    ax[idx].legend()
    print(f"temp={temp}")
    for q in [10, 25, 50, 75, 90]:
        print(f"{q}% Q={np.percentile(np.abs(np.log(q_p_ratio_ratios_temp)), q)}")

for idx, top_k in enumerate([5, 50, 150]):  # 150
    with open(
        Path(__file__).parent / "results" / f"top_k={top_k}_ratios_long.json", "r"
    ) as f:
        data = json.load(f)

    q_p_ratio_ratios_top_k = []
    for context in data:
        completion_ratios = [
            data[context]["q_ratios"][idx] / data[context]["p_ratios"][idx]
            for idx in range(len(data[context]["p_ratios"]))
        ]
        q_p_ratio_ratios_top_k.extend(completion_ratios)
    sns.ecdfplot(
        x=np.abs(np.log(q_p_ratio_ratios_top_k)),
        ax=ax[idx],
        label=f"Top-k: {top_k}",
        linewidth=5,
        color=sns.color_palette()[0],
        alpha=0.7,
    )

    print(f"top_k={top_k}")
    for q in [10, 25, 50, 75, 90]:
        print(f"{q}% Q={np.percentile(np.abs(np.log(q_p_ratio_ratios_top_k)), q)}")

for idx, hundred_top_p in enumerate([65, 88, 95]):
    top_p = hundred_top_p / 100
    with open(
        Path(__file__).parent / "results" / f"top_p={top_p}_ratios_long.json", "r"
    ) as f:
        data = json.load(f)

    q_p_ratio_ratios_top_p = []
    for context in data:
        completion_ratios = [
            data[context]["q_ratios"][idx] / data[context]["p_ratios"][idx]
            for idx in range(len(data[context]["p_ratios"]))
        ]
        q_p_ratio_ratios_top_p.extend(completion_ratios)
    sns.ecdfplot(
        x=np.abs(np.log(q_p_ratio_ratios_top_p)),
        ax=ax[idx],
        label=f"Top-p: {top_p}",
        linewidth=5,
        color=sns.color_palette()[3],
        alpha=0.6,
    )
    ax[idx].set_ylim([0, 1.1])
    print(f"top-p={top_p}")
    for q in [10, 25, 50, 75, 90]:
        print(f"{q}% Q={np.percentile(np.abs(np.log(q_p_ratio_ratios_top_p)), q)}")
    ax[idx].legend()


# ax[1].set_xlabel(
#     r"Log distortion ratio $|{\log{\left(p(\overline{w})q(\overline{v})\right) / \left(p(\overline{v})q(\overline{w})\right)}}|$",
#     fontsize=15,
# )
ax[1].set_xlabel(
    "Log distortion ratio",
    fontsize=15,
)
ax[0].set_ylabel("CDF", fontsize=15)

for a in ax:
    handles, labels = a.get_legend_handles_labels()
    a.legend(
        [handles[1], handles[2], handles[0]],
        [labels[1], labels[2], labels[0]],
        loc="lower right",
    )
    xmin, xmax = a.get_xlim()

fig.tight_layout()
fig.savefig(plot_dir / "measure_distortion_ratios.png")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), sharey=True, sharex=True)
