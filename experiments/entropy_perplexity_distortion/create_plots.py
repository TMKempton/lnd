import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

result_dir = Path("./experiments/entropy_perplexity_distortion/results")
plot_dir = Path("./experiments/entropy_perplexity_distortion/plots")
plot_dir.mkdir(exist_ok=True, parents=True)
norm_to_label = {"gn": "Global top-k", "ln": "Local top-k"}
type_to_palette = {"gn": "Blues", "ln": "Greens", "ps": "Reds"}

# Top-k subplot
ks = list(range(10, 101))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
entropies = []
perplexities = []
for i, norm_type in enumerate(["ln", "gn"]):
    entropies = []
    perplexities = []
    for k in ks:
        with open(result_dir / f"{norm_type}_top_k={k}_metrics.json", "r") as f:
            data = json.load(f)

        entropies.append(mean([data[context]["mean_entropy"] for context in data]))
        perplexities.append(
            mean([data[context]["mean_perplexity"] for context in data])
        )

    df = pd.DataFrame.from_dict(
        {"entropy": entropies, "perplexity": perplexities, f"{norm_type}_ks": ks}
    )
    sns.scatterplot(
        data=df,
        x="entropy",
        y="perplexity",
        ax=ax[0],
        label=norm_to_label[norm_type],
        color=sns.color_palette()[1 - i],
    )


# Top-p subplot
ps = [round(0.01 * x, 4) for x in range(40, 91)]
entropies = []
perplexities = []
for p in ps:
    with open(result_dir / f"ln_top_p={p}_metrics.json", "r") as f:
        data = json.load(f)
    entropies.append(mean([data[context]["mean_entropy"] for context in data]))
    perplexities.append(mean([data[context]["mean_perplexity"] for context in data]))

df = pd.DataFrame.from_dict(
    {"entropy": entropies, "perplexity": perplexities, "ps": ps}
)
sns.scatterplot(
    data=df,
    x="entropy",
    y="perplexity",
    ax=ax[1],
    label="Local top-p",
    color=sns.color_palette()[2],
)
entropies = []
perplexities = []
for p in ps:
    with open(result_dir / f"gn_top_p={p}_metrics.json", "r") as f:
        data = json.load(f)
    entropies.append(mean([data[context]["mean_entropy"] for context in data]))
    perplexities.append(mean([data[context]["mean_perplexity"] for context in data]))

df = pd.DataFrame.from_dict(
    {"entropy": entropies, "perplexity": perplexities, "ps": ps}
)
sns.scatterplot(
    data=df,
    x="entropy",
    y="perplexity",
    ax=ax[1],
    label="Global top-p",
    color=sns.color_palette()[3],
)


# Temp subplot
# temps = [round(0.01 * x, 4) for x in range(85, 101)]
# entropies = []
# perplexities = []

# for t in temps:
#     with open(result_dir / f"gn_temp={t}_metrics.json", "r") as f:
#         data = json.load(f)
#     entropies.append(mean([data[context]["mean_entropy"] for context in data]))
#     perplexities.append(mean([data[context]["mean_perplexity"] for context in data]))

# df = pd.DataFrame.from_dict(
#     {"entropy": entropies, "perplexity": perplexities, "temps": temps}
# )
# sns.scatterplot(
#     data=df,
#     x="entropy",
#     y="perplexity",
#     ax=ax[3],
#     label="Global temp",
#     color=sns.color_palette()[6],
# )

# entropies = []
# perplexities = []
# for t in temps:
#     with open(result_dir / f"ln_temp={t}_metrics.json", "r") as f:
#         data = json.load(f)
#     entropies.append(mean([data[context]["mean_entropy"] for context in data]))
#     perplexities.append(mean([data[context]["mean_perplexity"] for context in data]))

# df = pd.DataFrame.from_dict(
#     {"entropy": entropies, "perplexity": perplexities, "temps": temps}
# )
# sns.scatterplot(
#     data=df,
#     x="entropy",
#     y="perplexity",
#     ax=ax[3],
#     label="Local temp",
#     color=sns.color_palette()[5],
# )

ax[0].set_xlabel("")
ax[1].set_ylabel("")
ax[0].set_title("(a)", fontsize=15, pad=10)
ax[1].set_title("(b)", fontsize=15, pad=10)
ax[1].set_xlabel("", fontsize=15)
ax[0].set_ylabel("Negative log-likelihood", fontsize=15)
ax[1].legend()
sns.move_legend(ax[1], "upper left")
fig.supxlabel("Entropy", fontsize=15, ha="center")
fig.savefig(plot_dir / "entropy_perplexity_distortions.png", bbox_inches="tight")
