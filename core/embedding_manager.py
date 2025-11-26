import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION

MAX_EMBED_VALUE = 1e6  # Previne overflow de numeros

class EmbeddingManager:    
    def __init__(self):
        print(f"🔤 Loading embedding model: {EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.dimension = EMBEDDING_DIMENSION
        print(f"✅ Model loaded! Dimension: {self.dimension}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Cria embedding para um texto
           Retorna uma array Numpy com embeddings"""
        try:
            if not text or not text.strip():
                print("⚠️ Warning: Empty text provided for embedding")
                return np.zeros(self.dimension)
                
            embedding = self.embedding_model.encode(text)
            return np.array(embedding)
        
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return np.zeros(self.dimension)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Cria embeddings para varios textos"""
        if not texts:
            print("⚠️ No texts provided for batch embedding")
            return []
            
        embeddings = []
        total = len(texts)
        
        print(f"🔄 Generating {total} embeddings...")
        
        for i, text in enumerate(texts, 1):
            if i % 10 == 0 or i == total:
                print(f"  Processing embedding {i}/{total}...")
            
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def calculate_cosine_similarity(
        self, 
        vector1: np.ndarray, 
        vector2: np.ndarray
    ) -> float:
        """Calcula similaridade do coseno
           Retorna valor de similaridade entre 0 e 1"""
        # Lida com vetores invalidos
        if (not np.isfinite(vector1).all() or not np.isfinite(vector2).all() or
            np.all(vector1 == 0) or np.all(vector2 == 0)):
            return 0.0
        
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def calculate_batch_similarities(
        self,
        query_vector: np.ndarray,
        embeddings_matrix: np.ndarray
    ) -> np.ndarray:
        """Calcula similaridade do coseno entre um vetor e uma matrix de embeddings"""
        if len(embeddings_matrix) == 0:
            return np.array([])
            
        # Limpa e valida embeddins de entrada
        query_vector = np.nan_to_num(
            query_vector, nan=0.0, posinf=0.0, neginf=0.0
        )
        embeddings_matrix = np.nan_to_num(
            embeddings_matrix, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Limita valores menores ou maiores que o aceitavel
        query_vector = np.clip(query_vector, -MAX_EMBED_VALUE, MAX_EMBED_VALUE)
        embeddings_matrix = np.clip(embeddings_matrix, -MAX_EMBED_VALUE, MAX_EMBED_VALUE)

        # Calcula norma/magnitude
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0 or not np.isfinite(query_norm):
            return np.zeros(len(embeddings_matrix))
        
        # Calcula magnitudes de matrix
        chunk_norms = np.linalg.norm(embeddings_matrix, axis=1)
        # Prepara vetor vazio
        scores = np.zeros(len(embeddings_matrix))
        
        # Calcula apenas validos
        valid_indices = np.where(np.isfinite(chunk_norms) & (chunk_norms > 0))[0]
        if valid_indices.size == 0:
            return scores

        # Similaridade do cosseno
        denominator = chunk_norms[valid_indices] * query_norm
        numerator = embeddings_matrix[valid_indices] @ query_vector

        # Divisao segura
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator != 0
            )

        # Converte valores invalidos
        scores[valid_indices] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        # Retorno entre 0 e 1
        return np.clip(scores, 0.0, 1.0) 
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """Valida se o embedding e usavel"""
        return (isinstance(embedding, np.ndarray) and 
                embedding.shape == (self.dimension,) and 
                np.isfinite(embedding).all() and 
                not np.all(embedding == 0))
    
    def get_model_info(self) -> dict:
        """Rentorna informacoes sobre o modelo de embeddings"""
        return {
            'model_name': EMBEDDING_MODEL,
            'embedding_dimension': self.dimension,
            'max_sequence_length': getattr(self.embedding_model, 'max_seq_length', 512)
        }