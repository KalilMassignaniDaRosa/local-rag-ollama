from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    document_name: str
    text_content: str
    embedding: Optional[List[float]] = None
    page: Optional[int] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Comverte chunk em dicionario"""
        return {
            'chunk_id': self.chunk_id,
            'source_id': self.source_id,
            'document_name': self.document_name,
            'text_content': self.text_content,
            'page': self.page,
            'start_offset': self.start_offset,
            'end_offset': self.end_offset,
            'metadata': self.metadata
        }

@dataclass
class Source:
    source_id: str
    document_name: str
    chunk_id: str
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    page: Optional[int] = None
    excerpt: Optional[str] = None
    similarity_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte fonte em dicionario"""
        return {
            'source_id': self.source_id,
            'document_name': self.document_name,
            'chunk_id': self.chunk_id,
            'start_offset': self.start_offset,
            'end_offset': self.end_offset,
            'page': self.page,
            'excerpt': self.excerpt,
            'similarity_score': self.similarity_score
        }

@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List[Source]
    retrieved_chunks: int
    timestamp: str
    model: str
    latency_ms: Optional[float] = None
    
    def to_json(self) -> str:
        """Converte a resposta para JSON"""
        response_dict = {
            'question': self.question,
            'answer': self.answer,
            'sources': [source.to_dict() for source in self.sources],
            'retrieved_chunks': self.retrieved_chunks,
            'timestamp': self.timestamp,
            'model': self.model,
            'latency_ms': self.latency_ms
        }
        return json.dumps(response_dict, indent=2, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Coverte resposta para Markdown"""
        markdown = f"## Question\n{self.question}\n\n"
        markdown += f"## Answer\n{self.answer}\n\n"
        
        if self.sources:
            markdown += "## Sources\n"
            for source in self.sources:
                score_info = f", Score: {source.similarity_score:.4f}" if source.similarity_score else ""
                markdown += f"- {source.document_name} (Chunk: {source.chunk_id}{score_info})\n"
                if source.excerpt:
                    markdown += f"  > {source.excerpt[:200]}...\n"
        
        markdown += f"\n## Metadata\n"
        markdown += f"- Retrieved chunks: {self.retrieved_chunks}\n"
        markdown += f"- Model: {self.model}\n"
        if self.latency_ms:
            markdown += f"- Latency: {self.latency_ms:.2f}ms\n"
        markdown += f"- Timestamp: {self.timestamp}"
        
        return markdown
    
    def to_text(self) -> str:
        """Converte resposta em texto"""
        text = f"QUESTION: {self.question}\n\n"
        text += f"ANSWER: {self.answer}\n\n"
        
        if self.sources:
            text += f"SOURCES ({len(self.sources)}):\n"
            for i, source in enumerate(self.sources, 1):
                score_info = f" (Score: {source.similarity_score:.4f})" if source.similarity_score else ""
                text += f"{i}. {source.document_name} - Chunk: {source.chunk_id}{score_info}\n"
        
        text += f"\nMETADATA:\n"
        text += f"- Retrieved chunks: {self.retrieved_chunks}\n"
        text += f"- Model: {self.model}\n"
        if self.latency_ms:
            text += f"- Latency: {self.latency_ms:.2f}ms\n"
        text += f"- Timestamp: {self.timestamp}"
        
        return text

@dataclass
class SystemStatistics:
    total_chunks: int
    unique_documents: int
    ingested_documents: int
    embedding_dimension: int
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str
    
    def __str__(self):
        return f"""
            === RAG SYSTEM STATISTICS ===
            📊 Chunks in database: {self.total_chunks}
            📚 Unique documents: {self.unique_documents}
            📥 Ingested documents: {self.ingested_documents}
            🔢 Embedding dimension: {self.embedding_dimension}
            📏 Chunk size: {self.chunk_size}
            ↔️  Chunk overlap: {self.chunk_overlap}
            🎯 Chunking strategy: {self.chunking_strategy}
            ================================
            """
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte estatistica em dicionario"""
        return {
            'total_chunks': self.total_chunks,
            'unique_documents': self.unique_documents,
            'ingested_documents': self.ingested_documents,
            'embedding_dimension': self.embedding_dimension,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'chunking_strategy': self.chunking_strategy
        }
    
    def to_json(self) -> str:
        """Converte estatistica em JSON"""
        return json.dumps(self.to_dict(), indent=2)

# Funcoes para conveniencia
def create_chunk(
    chunk_id: str,
    source_id: str,
    document_name: str,
    text_content: str,
    **kwargs
) -> Chunk:
    """Fabrica chunks"""
    return Chunk(
        chunk_id=chunk_id,
        source_id=source_id,
        document_name=document_name,
        text_content=text_content,
        **kwargs
    )

def create_rag_response(
    question: str,
    answer: str,
    sources: List[Source],
    model: str,
    retrieved_chunks: int,
    latency_ms: Optional[float] = None
) -> RAGResponse:
    """Cria resposta RAG"""
    return RAGResponse(
        question=question,
        answer=answer,
        sources=sources,
        retrieved_chunks=retrieved_chunks,
        timestamp=datetime.now().isoformat(),
        model=model,
        latency_ms=latency_ms
    )