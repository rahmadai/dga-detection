import matplotlib.pyplot as plt
import numpy as np

# Data for visualization
categories = ["Accuracy", "Precision", "Recall", "F1 Score"]
semantic_only = [0.9499, 0.9394, 0.9307, 0.9200]
phonetic_only = [0.8061, 0.7347, 0.9433, 0.8336]

x = np.arange(len(categories))  # Label locations
width = 0.35  # Width of bars

# Creating the bar chart
fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, semantic_only, width, label="Semantic Only", color="blue", alpha=0.7)
bars2 = ax.bar(x + width/2, phonetic_only, width, label="Phonetic Only", color="red", alpha=0.7)

# Labels and title
ax.set_xlabel("Metrics")
ax.set_ylabel("Score")
ax.set_title("Single-Modality Baseline Comparison")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Display values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

# Show the plot
plt.ylim(0.7, 1.0)  # Set y-axis limits
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()