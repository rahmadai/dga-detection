import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from CSV
df = pd.read_csv("confusion_matrix_results.csv")

# Create a new column for total embedding size
df["Total Embedding Size"] = df["Phonetic Size"] + df["Semantic Size"]

# 1. Heatmap of Precision, Recall, and F1 Score
plt.figure(figsize=(12, 5))
metrics = ["Precision", "Recall", "F1 Score"]
for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 3, i)
    pivot_table = df.pivot(index="Phonetic Size", columns="Semantic Size", values=metric)
    sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".3f")
    plt.title(metric)
plt.tight_layout()
plt.show()

# 2. Line Plots for Performance Trends
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(df["Total Embedding Size"], df[metric], marker="o", label=metric)
plt.xlabel("Total Embedding Size")
plt.ylabel("Score")
plt.title("Performance Trends")
plt.legend()
plt.grid(True)
plt.show()

# 3. Bar Plot for TP, TN, FP, FN
plt.figure(figsize=(12, 6))
bars = ["TP", "TN", "FP", "FN"]
for bar in bars:
    plt.plot(df["Total Embedding Size"], df[bar], marker="o", label=bar)
plt.xlabel("Total Embedding Size")
plt.ylabel("Count")
plt.title("Confusion Matrix Components")
plt.legend()
plt.grid(True)
plt.show()

# 4. Bubble Chart for F1 Score
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df["Phonetic Size"], df["Semantic Size"], 
                       s=df["F1 Score"]*500, c=df["F1 Score"], cmap="coolwarm", alpha=0.7, edgecolors="k")
plt.colorbar(scatter, label="F1 Score")
plt.xlabel("Phonetic Size")
plt.ylabel("Semantic Size")
plt.title("Bubble Chart of F1 Score")
plt.grid(True)
plt.show()

# 5. Optimal Embedding Size Selection
plt.figure(figsize=(10, 6))
plt.plot(df["Total Embedding Size"], df["F1 Score"], marker="o", linestyle="--", color="b")
plt.xlabel("Total Embedding Size")
plt.ylabel("F1 Score")
plt.title("Optimal Embedding Size Analysis")
plt.grid(True)
plt.show()