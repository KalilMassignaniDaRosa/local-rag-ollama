import psycopg2
from psycopg2.extensions import register_adapter, AsIs
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from config.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

class DatabaseManager:    
    def __init__(self):
        self.db_config = {
            "host": DB_HOST,
            "port": DB_PORT,
            "database": DB_NAME,
            "user": DB_USER,
            "password": DB_PASSWORD
        }
        
        # Registra adaptador para numpy arrays
        self._register_vector_adapter()
        self._verify_connection()
        self._create_tables()
    
    def _register_vector_adapter(self):
        """Registra adaptador customizado para numpy arrays -> pgvector"""
        def adapt_numpy_array(array):
            """Converte numpy array para formato pgvector"""
            if isinstance(array, np.ndarray):
                array_list = array.tolist()
            else:
                array_list = list(array)
            # Retorna no formato: '[x,y,z]'
            return AsIs(f"'[{','.join(map(str, array_list))}]'::vector")
        
        # Registra o adaptador para numpy arrays
        register_adapter(np.ndarray, adapt_numpy_array)
    
    def _verify_connection(self):
        """Verifica conexao com banco"""
        try:
            conn = self.create_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            print(f"✅ Connected to PostgreSQL: {version[:50]}...")
            conn.close()
        except Exception as e:
            print(f"❌ Error connecting to PostgreSQL: {e}")
            raise
    
    def create_connection(self):
        """Cria nova conexao"""
        return psycopg2.connect(**self.db_config)
    
    def _create_tables(self):
        """Cria tabelas necessarias"""
        conn = self.create_connection()
        cursor = conn.cursor()
        
        # Cria extensao pgvector
        cursor.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
        """)
        
        # Cria chunks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(255) UNIQUE NOT NULL,
                source_id VARCHAR(255) NOT NULL,
                document_name TEXT NOT NULL,
                text_content TEXT NOT NULL,
                embedding vector(384),
                page INTEGER,
                start_offset INTEGER,
                end_offset INTEGER,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Cria index de busca
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding 
            ON document_chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        conn.commit()
        conn.close()
        print("✅ Tables created/verified successfully!")
    
    def add_chunk(
        self,
        chunk_id: str,
        source_id: str,
        document_name: str,
        text_content: str,
        embedding: np.ndarray,
        page: Optional[int] = None,
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """Adiciona novo chunk"""
        conn = self.create_connection()
        cursor = conn.cursor()
        
        # O adaptador registrado converte automaticamente
        cursor.execute(
            """
            INSERT INTO document_chunks 
            (chunk_id, source_id, document_name, text_content, embedding, 
             page, start_offset, end_offset, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                text_content = EXCLUDED.text_content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
            """,
            (chunk_id, source_id, document_name, text_content, embedding,
             page, start_offset, end_offset, json.dumps(metadata or {}))
        )
        
        conn.commit()
        conn.close()
    
    def add_chunks_batch(self, chunks_data: List[Dict]):
        """Adiciona varios chunks"""
        if not chunks_data:
            print("⚠️ No chunks to add")
            return
            
        conn = self.create_connection()
        cursor = conn.cursor()
        
        for chunk in chunks_data:
            # O adaptador registrado converte automaticamente
            cursor.execute(
                """
                INSERT INTO document_chunks 
                (chunk_id, source_id, document_name, text_content, embedding, 
                 page, start_offset, end_offset, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text_content = EXCLUDED.text_content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                (chunk['chunk_id'], chunk['source_id'], chunk['document_name'],
                 chunk['text_content'], chunk['embedding'], chunk.get('page'),
                 chunk.get('start_offset'), chunk.get('end_offset'),
                 json.dumps(chunk.get('metadata', {})))
            )
        
        conn.commit()
        conn.close()
        print(f"✅ {len(chunks_data)} chunks added to database!")
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Dict]:
        """Procura por chunks similares usando operador de distancia cosseno"""
        conn = self.create_connection()
        cursor = conn.cursor()
        
        # O adaptador registrado converte automaticamente o numpy array
        cursor.execute(
            """
            SELECT 
                chunk_id, source_id, document_name, text_content, 
                page, start_offset, end_offset, metadata,
                1 - (embedding <=> %s) as similarity_score
            FROM document_chunks
            WHERE 1 - (embedding <=> %s) >= %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, query_embedding, threshold, query_embedding, top_k)
        )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'chunk_id': row[0],
                'source_id': row[1],
                'document_name': row[2],
                'text_content': row[3],
                'page': row[4],
                'start_offset': row[5],
                'end_offset': row[6],
                'metadata': row[7],
                'similarity_score': float(row[8])
            })
        
        conn.close()
        return results
    
    def get_statistics(self) -> Dict:
        """Retorna estatisticas"""
        conn = self.create_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM document_chunks")
        total_chunks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT source_id) FROM document_chunks")
        unique_documents = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT document_name) FROM document_chunks")
        unique_document_names = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_chunks': total_chunks,
            'unique_documents': unique_documents,
            'unique_document_names': unique_document_names
        }
    
    def clean_database(self):
        """Remove todos os chunks"""
        conn = self.create_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM document_chunks")
        
        conn.commit()
        conn.close()
        print("✅ Database cleaned successfully!")
    
    def list_documents(self) -> List[Tuple[str, str, int]]:
        """Lista documentos com quantidades de chunks"""
        conn = self.create_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT source_id, document_name, COUNT(*) as chunk_count
            FROM document_chunks
            GROUP BY source_id, document_name
            ORDER BY source_id
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_chunks_by_document(self, document_name: str) -> List[Dict]:
        """Retorna todos os chunks de um documento"""
        conn = self.create_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT chunk_id, source_id, text_content, page, 
                   start_offset, end_offset, metadata
            FROM document_chunks
            WHERE document_name = %s
            ORDER BY page, start_offset
        """, (document_name,))
        
        chunks = []
        for row in cursor.fetchall():
            chunks.append({
                'chunk_id': row[0],
                'source_id': row[1],
                'text_content': row[2],
                'page': row[3],
                'start_offset': row[4],
                'end_offset': row[5],
                'metadata': row[6]
            })
        
        conn.close()
        return chunks