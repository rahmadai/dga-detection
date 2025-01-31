from sentence_transformers import SentenceTransformer, models
from sentence_transformers.models import StaticEmbedding
import torch
class SemanticPreprocessor:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        bge_static_embedding = StaticEmbedding.from_model2vec("FlukeTJ/bge-m3-m2v-distilled-256")
        self.semantic_model = SentenceTransformer(modules=[bge_static_embedding], device=device)
    
    def tokenize_domain(self, domain):
        """
        Generates the static embeddings for the input domain.
        Returns:
          - list of float: the fixed-size embedding for the domain.
        """
        # Encode the domain using the SentenceTransformer
        domain_embedding = self.semantic_model.encode(domain, convert_to_tensor=True)
        
        return domain_embedding.squeeze().tolist()