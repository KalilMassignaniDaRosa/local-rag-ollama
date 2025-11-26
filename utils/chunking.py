import re
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class ChunkingConfig:
    """Configuracao"""
    chunk_size: int = 800
    chunk_overlap: int = 150
    strategy: str = "fixed"

class ChunkingStrategy(ABC):    
    @abstractmethod
    def create_chunks(self, text: str) -> List[str]:
        """Divide texto em chunks baseado na estrategia especifica"""
        pass
    
    def _validate_input(self, text: str) -> None:
        """Valida texto de entrada"""
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string!")
        
        if len(text.strip()) == 0:
            raise ValueError("Input text cannot be empty or only whitespace!")

class FixedSizeChunking(ChunkingStrategy):
    """Estrategia de chunking com tamanho fixo e overlap"""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Valida parametros de configuracao do chunking"""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive!")
        
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative!")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size!")
    
    def create_chunks(self, text: str) -> List[str]:
        """Divide texto em chunks de tamanho fixo com overlap especificado"""
        self._validate_input(text)
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # So adiciona chunks nao vazios
            if chunk.strip():
                chunks.append(chunk)
            
            # Move para a proxima posicao de chunk com overlap
            start += self.chunk_size - self.chunk_overlap
            
            # Previne loop infinito com chunks muito pequenos
            if end >= text_length:
                break
        
        return chunks

class SemanticChunking(ChunkingStrategy):
    """Estrategia de chunking com consciencia semantica."""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Valida parametros de configuracao"""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive!")
        
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative!")
    
    def create_chunks(self, text: str) -> List[str]:
        """Divide texto em chunks preservando limites semanticos"""
        self._validate_input(text)
        
        # Divide por paragrafos (quebras de linha duplas)
        paragraphs = self._split_paragraphs(text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Se paragrafo for muito grande, divide por sentencas
            if len(paragraph) > self.chunk_size:
                sentences = self._split_sentences(paragraph)
                chunks.extend(self._process_sentences(sentences))
            else:
                # Tenta adicionar paragrafo ao chunk atual
                if self._can_add_to_chunk(current_chunk, paragraph):
                    current_chunk = self._append_to_chunk(current_chunk, paragraph)
                else:
                    # Finaliza chunk atual e comeca novo
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
        
        # Adiciona o ultimo chunk se existir
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Divide texto em paragrafos"""
        return re.split(r'\n\s*\n', text)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Divide texto em sentencas preservando abreviacoes"""
        # Divisao de sentencas melhorada que lida com abreviacoes comuns
        sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _process_sentences(self, sentences: List[str]) -> List[str]:
        """Processa sentencas em chunks respeitando limites de tamanho"""
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if self._can_add_to_chunk(current_chunk, sentence):
                current_chunk = self._append_to_chunk(current_chunk, sentence)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _can_add_to_chunk(self, current_chunk: str, new_text: str) -> bool:
        """Verifica se novo texto pode ser adicionado sem exceder  o limite"""
        if not current_chunk:
            return len(new_text) <= self.chunk_size
        
        combined_length = len(current_chunk) + len(new_text) + 1  # +1 para espaco
        return combined_length <= self.chunk_size
    
    def _append_to_chunk(self, current_chunk: str, new_text: str) -> str:
        """Adiciona novo texto ao chunk atual com espacamento apropriado"""
        if not current_chunk:
            return new_text
        return f"{current_chunk} {new_text}"

class SentenceAwareChunking(ChunkingStrategy):
    """Estrategia de chunking baseada em sentencas"""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def create_chunks(self, text: str) -> List[str]:
        """Divide texto em chunks que respeitam limites de sentencas"""
        self._validate_input(text)
        
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if self._can_add_to_chunk(current_chunk, sentence):
                current_chunk = self._append_to_chunk(current_chunk, sentence)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Manipula overlap para chunking baseado em sentencas
                if self.chunk_overlap > 0 and chunks:
                    last_chunk = chunks[-1]
                    overlap_sentences = self._get_overlap_sentences(last_chunk)
                    current_chunk = overlap_sentences + " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Divide texto em sentencas"""
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _can_add_to_chunk(self, current_chunk: str, sentence: str) -> bool:
        """Verifica se sentenca pode ser adicionada ao chunk atual"""
        if not current_chunk:
            return len(sentence) <= self.chunk_size
        
        combined_length = len(current_chunk) + len(sentence) + 1
        return combined_length <= self.chunk_size
    
    def _append_to_chunk(self, current_chunk: str, sentence: str) -> str:
        """Adiciona sentenca ao chunk atual"""
        if not current_chunk:
            return sentence
        return f"{current_chunk} {sentence}"
    
    def _get_overlap_sentences(self, chunk: str) -> str:
        """Obtem sentencas sobrepostas para continuidade entre chunks"""
        sentences = self._split_sentences(chunk)
        overlap_sentences = []
        overlap_length = 0
        
        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence) + 1
            else:
                break
        
        return " ".join(overlap_sentences)

class ChunkingFactory:
    """Classe factory para criar estrategias de chunking"""
    
    _strategies = {
        "fixed": FixedSizeChunking,
        "semantic": SemanticChunking,
        "sentence": SentenceAwareChunking,
    }
    
    @classmethod
    def create_strategy(
        cls, 
        strategy: str, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> ChunkingStrategy:
        """Cria uma instancia de estrategia de chunking"""
        strategy = strategy.lower()
        
        if strategy not in cls._strategies:
            available_strategies = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unsupported chunking strategy: '{strategy}'! "
                f"Available strategies: {available_strategies}"
            )
        
        strategy_class = cls._strategies[strategy]
        return strategy_class(chunk_size, chunk_overlap)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Obtem lista de estrategias de chunking disponiveis"""
        return list(cls._strategies.keys())

# Funcao de conveniencia para uso facil
def create_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    strategy: str = "fixed"
) -> List[str]:
    """Funcao para criar chunks a partir de texto"""
    chunker = ChunkingFactory.create_strategy(strategy, chunk_size, chunk_overlap)
    return chunker.create_chunks(text)