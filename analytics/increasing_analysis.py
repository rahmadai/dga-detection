import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the data
data = {
    'Phonetic Size': [64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512],
    'Semantic Size': [64, 128, 256, 512, 64, 128, 256, 512, 64, 128, 256, 512, 64, 128, 256, 512],
    'TP': [47915, 47865, 47965, 48165, 48265, 48215, 48466, 48566, 48666, 48766, 48816, 48916, 49016, 49116, 49216, 49266],
    'TN': [47238, 47396, 47495, 47591, 47690, 47794, 47889, 47989, 48088, 48188, 48238, 48337, 48285, 48334, 48281, 48331],
    'FP': [1945, 1787, 1688, 1592, 1493, 1389, 1294, 1194, 1095, 995, 945, 846, 898, 849, 902, 852],
    'FN': [2101, 2151, 2051, 1851, 1751, 1801, 1550, 1450, 1350, 1250, 1200, 1100, 1000, 900, 800, 750],
    'Precision': [0.9610, 0.9640, 0.9660, 0.9680, 0.9700, 0.9720, 0.9740, 0.9760, 0.9780, 0.9800, 0.9810, 0.9830, 0.9820, 0.9830, 0.9820, 0.9830],
    'Recall': [0.9580, 0.9570, 0.9590, 0.9630, 0.9650, 0.9640, 0.9690, 0.9710, 0.9730, 0.9750, 0.9760, 0.9780, 0.9800, 0.9820, 0.9840, 0.9850],
    'F1 Score': [0.9595, 0.9605, 0.9625, 0.9655, 0.9675, 0.9680, 0.9715, 0.9735, 0.9755, 0.9775, 0.9785, 0.9805, 0.9810, 0.9825, 0.9830, 0.9840]
}

# Create DataFrame
df = pd.DataFrame(data)

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')

# Set figure size
plt.figure(figsize=(15, 10))

#----------------------------------------------
# Plot 1: Performance with Increasing Phonetic Size
#----------------------------------------------
plt.subplot(2, 1, 1)

# Get unique phonetic and semantic sizes
phonetic_sizes = sorted(df['Phonetic Size'].unique())
semantic_sizes = sorted(df['Semantic Size'].unique())

# Create line plot for each semantic size
for semantic_size in semantic_sizes:
    # Filter data for this semantic size
    subset = df[df['Semantic Size'] == semantic_size]
    
    # Sort by phonetic size
    subset = subset.sort_values(by='Phonetic Size')
    
    # Plot F1 scores
    plt.plot(subset['Phonetic Size'], subset['F1 Score'], 
             marker='o', linewidth=2, label=f'Semantic Size = {semantic_size}')

# Add Precision and Recall for the largest semantic size (512)
semantic_size = 512
subset = df[df['Semantic Size'] == semantic_size].sort_values(by='Phonetic Size')
plt.plot(subset['Phonetic Size'], subset['Precision'], 
         marker='s', linestyle='--', linewidth=1.5, label=f'Precision (Semantic Size = {semantic_size})')
plt.plot(subset['Phonetic Size'], subset['Recall'], 
         marker='^', linestyle='--', linewidth=1.5, label=f'Recall (Semantic Size = {semantic_size})')

# Set axis labels and title
plt.xlabel('Phonetic Size', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Performance with Increasing Phonetic Size', fontsize=14)

# Set axis limits for better visualization
plt.ylim(0.955, 0.990)
plt.xticks(phonetic_sizes)

# Add legend
plt.legend(loc='lower right', fontsize=10)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

#----------------------------------------------
# Plot 2: Performance with Increasing Semantic Size
#----------------------------------------------
plt.subplot(2, 1, 2)

# Create line plot for each phonetic size
for phonetic_size in phonetic_sizes:
    # Filter data for this phonetic size
    subset = df[df['Phonetic Size'] == phonetic_size]
    
    # Sort by semantic size
    subset = subset.sort_values(by='Semantic Size')
    
    # Plot F1 scores
    plt.plot(subset['Semantic Size'], subset['F1 Score'], 
             marker='o', linewidth=2, label=f'Phonetic Size = {phonetic_size}')

# Add Precision and Recall for the largest phonetic size (512)
phonetic_size = 512
subset = df[df['Phonetic Size'] == phonetic_size].sort_values(by='Semantic Size')
plt.plot(subset['Semantic Size'], subset['Precision'], 
         marker='s', linestyle='--', linewidth=1.5, label=f'Precision (Phonetic Size = {phonetic_size})')
plt.plot(subset['Semantic Size'], subset['Recall'], 
         marker='^', linestyle='--', linewidth=1.5, label=f'Recall (Phonetic Size = {phonetic_size})')

# Set axis labels and title
plt.xlabel('Semantic Size', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Performance with Increasing Semantic Size', fontsize=14)

# Set axis limits for better visualization
plt.ylim(0.955, 0.990)
plt.xticks(semantic_sizes)

# Add legend
plt.legend(loc='lower right', fontsize=10)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig('performance_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

#----------------------------------------------
# Create a heatmap of F1 scores
#----------------------------------------------
plt.figure(figsize=(10, 8))

# Reshape data for heatmap
heatmap_data = df.pivot_table(values='F1 Score', index='Phonetic Size', columns='Semantic Size')

# Create heatmap
sns.heatmap(heatmap_data, annot=True, cmap='Blues', fmt='.4f', linewidths=.5, cbar_kws={'label': 'F1 Score'})

plt.title('F1 Score Heatmap by Phonetic and Semantic Size', fontsize=14)
plt.tight_layout()
plt.savefig('f1_score_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

#----------------------------------------------
# Additional Analysis: Performance gain with size increases
#----------------------------------------------
# Calculate the performance gain when increasing sizes
print("Performance Gains Analysis:")
print("=" * 50)

# Phonetic size gains (keeping semantic size constant)
print("\nF1 Score Gain when increasing Phonetic Size (for each Semantic Size):")
for sem_size in semantic_sizes:
    semantic_data = df[df['Semantic Size'] == sem_size].sort_values(by='Phonetic Size')
    base_f1 = semantic_data.iloc[0]['F1 Score']
    for i in range(1, len(semantic_data)):
        phonetic_size = semantic_data.iloc[i]['Phonetic Size']
        f1_score = semantic_data.iloc[i]['F1 Score']
        gain = f1_score - base_f1
        print(f"  Semantic Size {sem_size}: {semantic_data.iloc[0]['Phonetic Size']} → {phonetic_size}: +{gain:.4f}")
        base_f1 = f1_score

# Semantic size gains (keeping phonetic size constant)
print("\nF1 Score Gain when increasing Semantic Size (for each Phonetic Size):")
for phon_size in phonetic_sizes:
    phonetic_data = df[df['Phonetic Size'] == phon_size].sort_values(by='Semantic Size')
    base_f1 = phonetic_data.iloc[0]['F1 Score']
    for i in range(1, len(phonetic_data)):
        semantic_size = phonetic_data.iloc[i]['Semantic Size']
        f1_score = phonetic_data.iloc[i]['F1 Score']
        gain = f1_score - base_f1
        print(f"  Phonetic Size {phon_size}: {phonetic_data.iloc[0]['Semantic Size']} → {semantic_size}: +{gain:.4f}")
        base_f1 = f1_score

# Calculate and display the best configurations
best_row = df.loc[df['F1 Score'].idxmax()]
print("\nBest Overall Configuration:")
print(f"  Phonetic Size: {best_row['Phonetic Size']}, Semantic Size: {best_row['Semantic Size']}")
print(f"  F1 Score: {best_row['F1 Score']:.4f}, Precision: {best_row['Precision']:.4f}, Recall: {best_row['Recall']:.4f}")

# Find best cost-efficient configuration (at least 99% of the max F1 score)
max_f1 = df['F1 Score'].max()
threshold = 0.99 * max_f1
efficient_configs = df[df['F1 Score'] >= threshold].sort_values(by=['Phonetic Size', 'Semantic Size'])

if not efficient_configs.empty:
    efficient_row = efficient_configs.iloc[0]
    print("\nMost Efficient Configuration (99% of max F1 score):")
    print(f"  Phonetic Size: {efficient_row['Phonetic Size']}, Semantic Size: {efficient_row['Semantic Size']}")
    print(f"  F1 Score: {efficient_row['F1 Score']:.4f} ({efficient_row['F1 Score']/max_f1:.2%} of max)")
    print(f"  Precision: {efficient_row['Precision']:.4f}, Recall: {efficient_row['Recall']:.4f}")