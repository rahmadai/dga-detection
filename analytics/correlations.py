import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
np.random.seed(42)

# Sample domains and their features
n_samples = 100
n_semantic_features = 50
n_phonetic_features = 30

# Mock test domains
test_domains = [
    'google.com', 'facebook.com', 'amazon.com', 'netflix.com', 'microsoft.com'
] * 20  # Repeat to match n_samples

# Generate mock feature vectors
semantic_features = np.random.normal(0, 1, (n_samples, n_semantic_features))
phonetic_features = np.random.normal(0, 1, (n_samples, n_phonetic_features))

# Add some correlation between semantic and phonetic features
for i in range(min(n_semantic_features, n_phonetic_features)):
    phonetic_features[:, i] = 0.7 * semantic_features[:, i] + 0.3 * phonetic_features[:, i]

# Mock predictions (0 or 1 for binary classification)
true_labels = np.random.randint(0, 2, n_samples)
semantic_pred = (semantic_features.mean(axis=1) > 0).astype(int)
phonetic_pred = (phonetic_features.mean(axis=1) > 0).astype(int)

def analyze_modality_interactions(semantic_features, phonetic_features, test_domains, 
                                true_labels, semantic_pred, phonetic_pred):
    """
    Analyze interactions between semantic and phonetic features
    """
    results = {}
    
    # 1. Basic Correlation Analysis
    corr_matrix = np.corrcoef(semantic_features.T, phonetic_features.T)
    semantic_size = semantic_features.shape[1]
    cross_correlations = corr_matrix[:semantic_size, semantic_size:]
    results['cross_correlations'] = cross_correlations
    results['avg_correlation'] = np.mean(np.abs(cross_correlations))
    
    # 2. Mutual Information Analysis
    def discretize(x, bins=20):
        return np.digitize(x, np.linspace(x.min(), x.max(), bins))
    
    mi_scores = []
    for i in range(semantic_features.shape[1]):
        for j in range(phonetic_features.shape[1]):
            mi = mutual_info_score(
                discretize(semantic_features[:, i]),
                discretize(phonetic_features[:, j])
            )
            mi_scores.append(mi)
    
    results['mutual_information'] = {
        'mean': np.mean(mi_scores),
        'std': np.std(mi_scores)
    }
    
    # 3. Complementary Analysis
    semantic_correct = semantic_pred == true_labels
    phonetic_correct = phonetic_pred == true_labels
    
    complementary_cases = {
        'semantic_only': np.where(semantic_correct & ~phonetic_correct)[0],
        'phonetic_only': np.where(phonetic_correct & ~semantic_correct)[0],
        'both_correct': np.where(semantic_correct & phonetic_correct)[0],
        'both_wrong': np.where(~semantic_correct & ~phonetic_correct)[0]
    }
    
    results['complementary_cases'] = {
        key: {
            'count': len(indices),
            'examples': [test_domains[i] for i in indices[:5]]  # First 5 examples
        }
        for key, indices in complementary_cases.items()
    }
    
    # 4. Dimensionality Analysis
    pca = PCA()
    semantic_pca = pca.fit(semantic_features)
    phonetic_pca = pca.fit(phonetic_features)
    
    results['dimensionality'] = {
        'semantic_explained_variance': semantic_pca.explained_variance_ratio_,
        'phonetic_explained_variance': phonetic_pca.explained_variance_ratio_
    }
    
    return results

def visualize_results(results):
    """
    Create visualizations for the analysis results
    """
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['cross_correlations'], 
                cmap='RdBu', 
                center=0,
                xticklabels=False, 
                yticklabels=False)
    plt.title('Semantic-Phonetic Feature Correlations')
    plt.xlabel('Phonetic Features')
    plt.ylabel('Semantic Features')
    plt.show()
    
    # 2. Explained Variance Plot
    plt.figure(figsize=(10, 4))
    plt.plot(np.cumsum(results['dimensionality']['semantic_explained_variance']), 
             label='Semantic')
    plt.plot(np.cumsum(results['dimensionality']['phonetic_explained_variance']), 
             label='Phonetic')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Dimensionality Analysis')
    plt.legend()
    plt.show()

# Run the analysis
results = analyze_modality_interactions(
    semantic_features,
    phonetic_features,
    test_domains,
    true_labels,
    semantic_pred,
    phonetic_pred
)

# Print key findings
print("\nKey Findings:")
print(f"Average Feature Correlation: {results['avg_correlation']:.3f}")
print(f"Average Mutual Information: {results['mutual_information']['mean']:.3f}")
print("\nComplementary Analysis:")
for case, data in results['complementary_cases'].items():
    print(f"\n{case}:")
    print(f"Count: {data['count']}")
    print(f"Example domains: {', '.join(data['examples'])}")

# Create visualizations
visualize_results(results)