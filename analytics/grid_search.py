import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("compare.csv")

# Define colors
colors = ['blue', 'red', 'orange', 'green']
metrics = ["precision", "recall", "f1"]
titles = ["Precision", "Recall", "F1"]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(metrics):
    for idx, embed_size in enumerate(df["phonetic_embedding_size"].unique()):
        subset = df[df["phonetic_embedding_size"] == embed_size]
        axes[i].plot(subset["semantic_embedding_size"], subset[metric], 
                     label=f"E-Phonetic={embed_size}", color=colors[idx], marker='o')

    axes[i].set_xlabel("Semantic Embedding Size")
    axes[i].set_ylabel(metric.capitalize())
    axes[i].set_title(titles[i])
    axes[i].legend()
    axes[i].set_xticks([64, 128, 256, 512])

plt.tight_layout()
plt.show()
