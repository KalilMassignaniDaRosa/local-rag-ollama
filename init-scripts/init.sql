-- PostgreSQL database setup script for RAG

-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create chunks table
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

-- Create index for efficient vector search
CREATE INDEX IF NOT EXISTS idx_embedding 
ON document_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for source search
CREATE INDEX IF NOT EXISTS idx_source_id 
ON document_chunks(source_id);

-- Create index for document search
CREATE INDEX IF NOT EXISTS idx_document_name 
ON document_chunks(document_name);

-- Check if everything was created
SELECT 
    'Vector extension installed' AS status
FROM pg_extension 
WHERE extname = 'vector';

SELECT 
    'document_chunks table created' AS status,
    COUNT(*) AS total_chunks
FROM document_chunks;
