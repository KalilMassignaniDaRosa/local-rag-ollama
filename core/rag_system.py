import ollama
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

class SimpleRAGSystem:    
    def __init__(self, db_config: Optional[Dict] = None):
        self.db_config = db_config or {
            "host": "localhost",
            "port": "5433",
            "database": "rag_database",
            "user": "postgres",
            "password": "postgres"
        }
        
        print("🔤 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384 
        
        self.llm_model = "llama3.2:3b"
        
        # Inicializa banco de dados
        self._initialize_database()
        print("✅ Simple RAG System initialized!")
    
    def _initialize_database(self):
        """Inicializa conexao com banco e cria tabelas necessarias"""
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            
            # Cria extensao se nao existir
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Cria indice para busca vetorial
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_embeddings 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops);
            """)
            
            conn.commit()
            conn.close()
            print("✅ Database tables verified/created")
            
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            raise
    
    def create_connection(self):
        """Cria conexao com banco de dados"""
        return psycopg2.connect(**self.db_config)
    
    def add_document(self, text: str, metadata: Dict = None) -> bool:
        """Adiciona documento ao banco de dados"""
        if not text or not text.strip():
            print("⚠️  Empty text provided")
            return False
            
        try:
            # Gera embedding
            embedding = self.embedding_model.encode(text)
            embedding_list = embedding.tolist()
            
            # Conecta e insere no banco
            conn = self.create_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                INSERT INTO document_chunks (content, embedding, metadata)
                VALUES (%s, %s, %s)
                """,
                (text.strip(), embedding_list, json.dumps(metadata or {}))
            )
            
            conn.commit()
            conn.close()
            print(f"✅ Document added: {len(text)} characters")
            return True
            
        except Exception as e:
            print(f"❌ Error adding document: {e}")
            return False
    
    def add_documents_batch(self, documents: List[str], metadata_list: List[Dict] = None):
        """Adiciona varios documentos de uma vez"""
        if not documents:
            print("⚠️  No documents provided")
            return
            
        if metadata_list and len(metadata_list) != len(documents):
            print("⚠️  Metadata list length doesn't match documents list")
            metadata_list = None
            
        success_count = 0
        for i, doc in enumerate(documents):
            metadata = metadata_list[i] if metadata_list else None
            if self.add_document(doc, metadata):
                success_count += 1
        
        print(f"📊 Added {success_count}/{len(documents)} documents successfully")
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        """Procura por documentos similares"""
        if not query or not query.strip():
            print("⚠️  Empty query provided")
            return []
            
        try:
            # Gera embedding da consulta
            query_embedding = self.embedding_model.encode(query)
            query_embedding_list = query_embedding.tolist()
            
            # Procura documentos similares
            conn = self.create_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT content, metadata, 
                       embedding <=> %s as distance
                FROM document_chunks 
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding_list, query_embedding_list, top_k)
            )
            
            results = []
            for content, metadata, distance in cursor.fetchall():
                results.append({
                    "content": content,
                    "metadata": metadata,
                    # Converte distancia para similaridade
                    "similarity": 1 - distance,  
                    "distance": distance
                })
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"❌ Error searching similar documents: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Retorna numero total de documentos no banco"""
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
        except Exception as e:
            print(f"❌ Error getting document count: {e}")
            return 0
    
    def ask_question(self, question: str, top_k: int = 3) -> str:
        """Faz pergunta ao sistema RAG"""
        if not question or not question.strip():
            return "Please provide a valid question."
        
        print(f"🔍 Searching for relevant contexts...")
        
        # Procura contextos similares
        similar_docs = self.search_similar(question, top_k)
        
        if not similar_docs:
            return "I couldn't find relevant information to answer your question in the available documents."
        
        # Cria contexto com documentos similares
        context = "\n\n".join([
            f"[Document {i+1}]: {doc['content']}" 
            for i, doc in enumerate(similar_docs)
        ])
        
        # Cria prompt para LLM
        prompt = f"""Based on the following documents, please answer the question accurately and concisely.

            Relevant Documents:
            {context}

            Question: {question}

            Instructions:
            - Base your answer only on the provided documents
            - Be concise and direct
            - If the documents don't contain sufficient information, say "The available documents don't contain enough information to answer this question"
            - Cite relevant document numbers in your response when appropriate

            Answer:"""
        
        print(f"🧠 Generating response using {len(similar_docs)} contexts...")
        
        try:
            # Gera resposta usando Ollama
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    # Mais deterministico
                    'temperature': 0.1,
                    'top_p': 0.9
                }
            )
            
            return response['response']
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def clear_documents(self):
        """Remove todos os documentos do banco"""
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM document_chunks")
            conn.commit()
            conn.close()
            
            print("✅ All documents cleared from database")
        except Exception as e:
            print(f"❌ Error clearing documents: {e}")


def main():
    rag = SimpleRAGSystem()
    
    # Exemplos de documentos
    documents = [
        "PostgreSQL is an open-source object-relational database system.",
        "Ollama is a tool for running large language models locally.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation.",
        "pgvector is a PostgreSQL extension for storing and searching vector embeddings.",
        "Sentence Transformers is a framework for state-of-the-art sentence and text embeddings."
    ]
    
    print("📝 Adding example documents...")
    for i, doc in enumerate(documents):
        rag.add_document(doc, {"source": "example", "id": i})
    
    print(f"📊 Total documents in database: {rag.get_document_count()}")
    
    # Teste de perguntas
    test_questions = [
        "What is PostgreSQL?",
        "How does RAG work?",
        "What is pgvector used for?"
    ]
    
    print("\n🎯 Testing RAG system with sample questions:")
    print("-" * 50)
    
    for question in test_questions:
        print(f"\n🤔 Question: {question}")
        answer = rag.ask_question(question)
        print(f"🧠 Answer: {answer}")
        print("-" * 50)


if __name__ == "__main__":
    main()