import os
import pathlib
import re
from datetime import datetime
from typing import List, Dict, Optional, Union
import PyPDF2

from config.settings import (
    DOCUMENTS_FOLDER, DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD
)

from core.models import (
    Chunk, RAGResponse, Source, SystemStatistics
)

from core.database_manager import DatabaseManager
from core.embedding_manager import EmbeddingManager
from core.llm_manager import LLMManager
from utils.chunking import ChunkingFactory
from utils.prompt_templates import PromptTemplates

class AdvancedRAGSystem:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        chunking_strategy: str = DEFAULT_CHUNKING_STRATEGY
    ):
        print("🚀 Initializing Advanced RAG System...")
        
        self.db = DatabaseManager()
        self.embeddings = EmbeddingManager()
        self.llm = LLMManager()
        
        # Configuracao de chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.strategy = ChunkingFactory.create_strategy(
            chunking_strategy, 
            chunk_size, 
            chunk_overlap
        )
        
        # Armazena documentos originais
        self.original_documents: List[Dict] = []
        
        # Configuracao de pastas
        self.documents_folder = DOCUMENTS_FOLDER
        self._create_folders()
        
        print("✅ System initialized successfully!")
    
    def _create_folders(self):
        """Cria pastas necessarias se nao existirem"""
        if not os.path.exists(self.documents_folder):
            os.makedirs(self.documents_folder)
            print(f"📁 Folder '{self.documents_folder}' created!")
    
    # === Ingestao de Documentos ===
    def ingest_pdf(self, pdf_path: str) -> bool:
        """Ingere um unico PDF"""
        filepath = pathlib.Path(pdf_path)
        
        if not filepath.exists():
            print(f"❌ File not found: {pdf_path}")
            return False
        
        try:
            source_id = f"doc_{len(self.original_documents) + 1}"
            
            self.original_documents.append({
                'source_id': source_id,
                'name': filepath.name,
                'path': str(pdf_path),
                'data': filepath.read_bytes(),
                'ingested_at': datetime.now().isoformat()
            })
            
            print(f"✅ PDF ingested: {filepath.name} [ID: {source_id}]")
            return True
            
        except Exception as e:
            print(f"❌ Error during ingestion: {e}")
            return False
    
    def ingest_all_pdfs(self) -> bool:
        """Ingere todos os PDFs da pasta de documentos"""
        pdfs = [
            f for f in os.listdir(self.documents_folder) 
            if f.lower().endswith('.pdf')
        ]
        
        if not pdfs:
            print(f"❌ No PDFs found in '{self.documents_folder}'")
            return False
        
        print(f"\n📥 Ingesting {len(pdfs)} PDF(s)...")
        for pdf in pdfs:
            self.ingest_pdf(os.path.join(self.documents_folder, pdf))
        
        return True
    
    # === Processamento e Indexacao ===
    def process_and_index(
        self, 
        document_index: Optional[int] = None
    ) -> bool:
        """Processa documentos e cria embeddings"""
        if not self.original_documents:
            print("❌ No documents ingested. Use ingest_pdf() first.")
            return False
        
        # Seleciona documentos para processar
        docs_to_process = (
            [self.original_documents[document_index]] 
            if document_index is not None 
            else self.original_documents
        )
        
        print(f"\n📊 Processing {len(docs_to_process)} document(s)...\n")
        
        for doc in docs_to_process:
            print(f"⏳ Processing: {doc['name']}")
            
            # Extracao de texto
            print("  [1/4] Extracting text from PDF...")
            full_text = self._extract_pdf_text(doc['data'])
            
            if not full_text:
                print("  ❌ Text extraction failed")
                continue
            
            # Criacao de chunks
            print(f"  [2/4] Creating chunks (strategy: {self.chunking_strategy})...")
            text_chunks = self.strategy.create_chunks(full_text)
            print(f"  ✓ {len(text_chunks)} chunks created")
            
            # Geracao de embeddings
            print("  [3/4] Generating embeddings...")
            embeddings = self.embeddings.generate_embeddings_batch(text_chunks)
            print(f"  ✓ {len(embeddings)} embeddings generated")
            
            # Salva no banco de dados
            print("  [4/4] Saving to PostgreSQL...")
            chunks_data = self._create_chunks_data(doc, text_chunks, embeddings)
            self.db.add_chunks_batch(chunks_data)
            print(f"  ✓ {len(chunks_data)} chunks saved\n")
        
        statistics = self.get_statistics()
        print(f"✅ Processing completed!")
        print(statistics)
        
        return True
    
    def _extract_pdf_text(self, pdf_data: bytes) -> str:
        """Extrai texto de PDF usando PyPDF2"""
        try:
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
            
            return text.strip()
        except Exception as e:
            print(f"  ⚠️  Error extracting text: {e}")
            return ""
    
    def _create_chunks_data(
        self, 
        document: Dict, 
        text_chunks: List[str],
        embeddings: List
    ) -> List[Dict]:
        """Cria dados estruturados para armazenamento no banco"""
        chunks_data = []
        
        for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
            chunk_id = f"{document['source_id']}_chunk_{i+1}"
            
            chunks_data.append({
                'chunk_id': chunk_id,
                'source_id': document['source_id'],
                'document_name': document['name'],
                'text_content': chunk_text,
                'embedding': embedding,
                'start_offset': i * (self.chunk_size - self.chunk_overlap),
                'end_offset': (i + 1) * self.chunk_size,
                'metadata': {'strategy': self.chunking_strategy}
            })
        
        return chunks_data
    
    # === Consulta RAG ===
    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        template_type: str = "query",
        output_format: str = "text"  # text, json, markdown
    ) -> Union[RAGResponse, str]:
        """Executa consulta RAG completa"""
        start_time = datetime.now()
        
        stats = self.db.get_statistics()
        if stats['total_chunks'] == 0:
            return self._create_error_response(
                question,
                "❌ Empty database. Execute process_and_index() first."
            )
        
        try:
            # [1/3] Gera embedding da pergunta
            print(f"🔍 Generating question embedding...")
            question_embedding = self.embeddings.generate_embedding(question)
            
            # [2/3] Recuperacao vetorial
            print(f"🔍 Searching chunks (top_k={top_k}, threshold={threshold})...")
            db_results = self.db.search_similar(
                question_embedding, 
                top_k, 
                threshold
            )
            
            if not db_results:
                return self._create_error_response(
                    question,
                    "⚠️  No relevant chunks found. Try lowering the threshold."
                )
            
            print(f"✓ {len(db_results)} chunks retrieved")
            
            # [3/3] Geracao de resposta
            print("💬 Generating response with Ollama...")
            prompt = self._select_template(template_type, question, db_results)
            
            response_text = self.llm.generate_response(prompt)
            
            # Extrai fontes citadas
            sources = self._extract_sources(response_text, db_results)
            
            # Calcula latencia
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            rag_response = RAGResponse(
                question=question,
                answer=response_text,
                sources=sources,
                retrieved_chunks=len(db_results),
                timestamp=datetime.now().isoformat(),
                model=self.llm.model,
                latency_ms=latency
            )
            
            print(f"✅ Response generated in {latency:.2f}ms\n")
            
            # Retorna no formato solicitado
            if output_format == "json":
                return rag_response.to_json()
            elif output_format == "markdown":
                return rag_response.to_markdown()
            else:
                return rag_response
            
        except Exception as e:
            return self._create_error_response(question, f"❌ Error: {e}")
    
    def _select_template(self, template_type: str, question: str, results):
        """Seleciona template de prompt apropriado"""
        template_type = template_type.lower()
        
        if template_type == "summary":
            return PromptTemplates.document_summary_template(results)
        elif template_type in ("comparison", "comparacao"):
            return PromptTemplates.comparison_template(question, results)
        elif template_type in ("extraction", "extracao"):
            return PromptTemplates.information_extraction_template(question, results)
        elif template_type == "qa":
            return PromptTemplates.question_answer_template(question, results)
        else:
            return PromptTemplates.advanced_query_template(question, results)
    
    def _extract_sources(self, response_text: str, results) -> List[Source]:
        """Extrai fontes citadas na resposta"""
        # Mapa de chunks para acesso rapido
        chunk_map = {
            r['chunk_id']: r 
            for r in results
        }
        
        # Extrai IDs de chunks citados
        cited_ids = set(re.findall(r'\[(doc_\d+_chunk_\d+)\]', response_text))
        
        sources = []
        for chunk_id in cited_ids:
            if chunk_id in chunk_map:
                chunk_data = chunk_map[chunk_id]
                
                sources.append(Source(
                    source_id=chunk_data['source_id'],
                    document_name=chunk_data['document_name'],
                    chunk_id=chunk_data['chunk_id'],
                    start_offset=chunk_data.get('start_offset'),
                    end_offset=chunk_data.get('end_offset'),
                    page=chunk_data.get('page'),
                    excerpt=chunk_data['text_content'][:200],
                    similarity_score=chunk_data['similarity_score']
                ))
        
        return sources
    
    def _create_error_response(self, question: str, message: str) -> RAGResponse:
        """Cria resposta de erro"""
        return RAGResponse(
            question=question,
            answer=message,
            sources=[],
            retrieved_chunks=0,
            timestamp=datetime.now().isoformat(),
            model=self.llm.model
        )
    
    # === Utilitarios do Sistema ===
    def get_statistics(self) -> SystemStatistics:
        """Retorna estatisticas do sistema"""
        db_stats = self.db.get_statistics()
        
        return SystemStatistics(
            total_chunks=db_stats['total_chunks'],
            unique_documents=db_stats['unique_documents'],
            ingested_documents=len(self.original_documents),
            embedding_dimension=self.embeddings.dimension,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunking_strategy=self.chunking_strategy
        )
    
    def list_documents(self) -> str:
        """Lista todos os documentos indexados"""
        docs = self.db.list_documents()
        
        if not docs:
            return "📚 No documents in database."
        
        lines = [
            f"  [{source_id}] {name} ({chunks} chunks)" 
            for source_id, name, chunks in docs
        ]
        
        return "📚 Documents in Database:\n" + "\n".join(lines)
    
    def clean_system(self):
        """Limpa todo o sistema (banco e memoria)"""
        self.db.clean_database()
        self.original_documents = []
        print("✅ System cleaned")
    
    def get_document_info(self, document_name: str) -> Dict:
        """Obtem informacoes detalhadas de um documento"""
        chunks = self.db.get_chunks_by_document(document_name)
        
        if not chunks:
            return {"error": f"Document '{document_name}' not found"}
        
        return {
            'document_name': document_name,
            'total_chunks': len(chunks),
            'chunks': chunks[:10],  # Primeiros 10 chunks
            'sample_text': chunks[0]['text_content'][:500] if chunks else ""
        }