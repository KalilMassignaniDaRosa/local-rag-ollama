import os
import sys
import traceback

# Adiciona diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.advanced_rag_system import AdvancedRAGSystem
from config.settings import (
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNKING_STRATEGY, DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD
)

MENU_SECTIONS = [
    ("📥 INGESTION", [
        ("1", "Ingest all PDFs from folder")
    ]),
    ("🔧 PROCESSING", [
        ("2", "Process and index documents (chunking + embeddings)")
    ]),
    ("🔍 QUERIES", [
        ("3", "Query index (simple mode)"),
        ("4", "Query with advanced settings")
    ]),
    ("📊 INFORMATION", [
        ("5", "View system statistics"),
        ("6", "List indexed documents"),
        ("7", "Run system diagnostic")
    ]),
    ("🧹 MAINTENANCE", [
        ("8", "Clean system")
    ]),
    ("❌ EXIT", [
        ("0", "Shutdown system")
    ])
]

class RagCLI:
    def __init__(self):
        self.system = None
        self.actions = {
            "1": self._action_ingest,
            "2": self._action_process,
            "3": self._action_query_simple,
            "4": self._action_query_advanced,
            "5": self._action_statistics,
            "6": self._action_list_documents,
            "7": self._action_diagnostic,
            "8": self._action_clean_system
        }
    
    def clear_screen(self):
        """Limpa a tela do terminal"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def wait_for_user(self, message: str = "\n⏸️  Press Enter to continue..."):
        """Espera usuario pressionar Enter"""
        input(message)
    
    def get_user_input(
        self,
        message: str,
        default_value=None,
        conversion_type=str
    ):
        """Obtem entrada do usuario com valor padrao"""
        hint = f" [{default_value}]" if default_value is not None else ""
        user_input = input(f"{message}{hint}: ").strip()
        
        if not user_input and default_value is not None:
            return default_value
        
        try:
            return conversion_type(user_input)
        except ValueError:
            print(f"⚠️  Invalid value. Using default: {default_value}")
            return default_value
    
    def show_menu(self):
        """Mostra menu principal"""
        print("\n" + "=" * 70)
        print("🚀 Advanced Local RAG System - Main Menu")
        print("=" * 70)
        
        for title, options in MENU_SECTIONS:
            print(f"\n{title}")
            for code, description in options:
                print(f"  {code}. {description}")
        
        print("\n" + "=" * 70)
    
    def _initialize_system(self):
        """Inicializa sistema com configuracao"""
        print("\n⚙️  System Initial Configuration")
        print("-" * 70)
        
        chunk_size = self.get_user_input(
            "📏 Chunk size (characters)",
            DEFAULT_CHUNK_SIZE,
            int
        )
        
        overlap = self.get_user_input(
            "↔️  Chunk overlap (characters)",
            DEFAULT_CHUNK_OVERLAP,
            int
        )
        
        print("\n🎯 Available Chunking Strategies:")
        print("  - fixed: Fixed size with overlap")
        print("  - semantic: Paragraph-based")
        print("  - sentence: Sentence-based")
        
        strategy = self.get_user_input(
            "\n🎯 Chunking strategy",
            DEFAULT_CHUNKING_STRATEGY
        )
        
        # Inicializa sistema
        print("\n🔄 Initializing RAG System...")
        self.system = AdvancedRAGSystem(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            chunking_strategy=strategy
        )
        
        print(f"\n📊 Settings: chunk={chunk_size}, overlap={overlap}, strategy={strategy}")
        self.wait_for_user("\n⏸️  Press Enter to access menu...")
    
    # === Acoes do menu ===
    def _action_ingest(self):
        """Ingestao de documentos"""
        print("\n" + "=" * 70)
        print("📥 Document Ingestion")
        print("=" * 70)
        
        self.system.ingest_all_pdfs()
        self.wait_for_user()
    
    def _action_process(self):
        """Processa e indexa documentos"""
        if not self.system.original_documents:
            print("\n❌ No documents ingested. Execute option 1 first.")
            self.wait_for_user()
            return
        
        print("\n" + "=" * 70)
        print("🔧 Document Processing")
        print("=" * 70)
        
        self.system.process_and_index()
        self.wait_for_user()
    
    def _action_query_simple(self):
        """Consulta simples"""
        stats = self.system.get_statistics()
        if stats.total_chunks == 0:
            print("\n❌ Empty database. Execute option 2 first.")
            self.wait_for_user()
            return
        
        print("\n" + "=" * 70)
        print("🔍 Simple Query")
        print("=" * 70)
        
        question = input("\n❓ Enter your question: ").strip()
        if not question:
            print("❌ Empty question.")
            self.wait_for_user()
            return
        
        output_format = self.get_user_input(
            "📄 Output format (text/json/markdown)",
            "text"
        ).lower()
        
        # Dica sobre threshold
        print("\n💡 Tip: If no results, try option 4 with lower threshold (<0.5)")
        print(f"   Current default threshold: {DEFAULT_SIMILARITY_THRESHOLD}")
        
        print("\n⏳ Processing query...")
        response = self.system.query(question, output_format=output_format)
        
        self._display_response(response, output_format)
        self.wait_for_user()
    
    def _action_query_advanced(self):
        """Consulta avancada"""
        stats = self.system.get_statistics()
        if stats.total_chunks == 0:
            print("\n❌ Empty database. Execute option 2 first.")
            self.wait_for_user()
            return
        
        print("\n" + "=" * 70)
        print("🔍 Advanced Query")
        print("=" * 70)
        
        question = input("\n❓ Enter your question: ").strip()
        if not question:
            print("❌ Empty question!")
            self.wait_for_user()
            return
        
        print("\n⚙️  Advanced Settings")
        top_k = self.get_user_input(
            "📊 Top K (number of chunks to retrieve)",
            DEFAULT_TOP_K,
            int
        )
        
        threshold = self.get_user_input(
            "🎯 Threshold (minimum similarity score, 0-1)",
            DEFAULT_SIMILARITY_THRESHOLD,
            float
        )
        
        print("\n🎨 Available Template Types:")
        print("  - query: Standard query with citations")
        print("  - summary: Generate structured summary")
        print("  - comparison: Comparative analysis")
        print("  - extraction: Specific information extraction")
        print("  - qa: Optimized question-answering")
        
        template_type = self.get_user_input(
            "\n🎨 Template type",
            "query"
        ).lower()
        
        output_format = self.get_user_input(
            "📄 Output format (text/json/markdown)",
            "text"
        ).lower()
        
        print("\n⏳ Processing advanced query...")
        response = self.system.query(
            question,
            top_k=top_k,
            threshold=threshold,
            template_type=template_type,
            output_format=output_format
        )
        
        self._display_response(response, output_format, {
            'top_k': top_k,
            'threshold': threshold,
            'template_type': template_type
        })
        self.wait_for_user()
    
    def _display_response(self, response, output_format: str, advanced_metrics: dict = None):
        """Mostra resposta no formato correto"""
        if output_format == "text" and hasattr(response, 'answer'):
            print("\n" + "=" * 70)
            print("💡 Response")
            print("=" * 70)
            print(f"\n{response.answer}\n")
            
            if response.sources:
                print("=" * 70)
                print(f"📚 Cited sources ({len(response.sources)})")
                print("=" * 70)
                for i, source in enumerate(response.sources, 1):
                    print(f"\n[{i}] {source.document_name}")
                    print(f"    Chunk: {source.chunk_id}")
                    if source.similarity_score:
                        print(f"    Score: {source.similarity_score:.4f}")
            
            print("\n" + "=" * 70)
            print("📊 Metrics")
            print("=" * 70)
            print(f"  Retrieved chunks: {response.retrieved_chunks}")
            if response.latency_ms:
                print(f"  Latency: {response.latency_ms:.2f}ms")
            print(f"  Model: {response.model}")
            
            if advanced_metrics:
                for key, value in advanced_metrics.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        else:
            print("\n" + "=" * 70)
            print(f"📄 Response ({output_format.upper()})")
            print("=" * 70)
            print(f"\n{response}")
    
    def _action_statistics(self):
        """Mostra estatisticas do sistema"""
        print("\n" + "=" * 70)
        stats = self.system.get_statistics()
        print(stats)
        print("=" * 70)
        self.wait_for_user()
    
    def _action_list_documents(self):
        """Lista documentos indexados"""
        print("\n" + "=" * 70)
        print(self.system.list_documents())
        print("=" * 70)
        self.wait_for_user()
    
    def _action_diagnostic(self):
        """Executa diagnóstico completo do sistema"""
        print("\n" + "=" * 70)
        print("🔍 System Diagnostic")
        print("=" * 70)
        
        # Estatísticas básicas
        stats = self.system.get_statistics()
        print(f"\n📊 Database Statistics:")
        print(f"  Total chunks: {stats.total_chunks}")
        print(f"  Unique documents: {stats.unique_documents}")
        print(f"  Ingested documents: {stats.ingested_documents}")
        
        if stats.total_chunks == 0:
            print("\n❌ No chunks in database!")
            print("   Run option 1 (Ingest) then option 2 (Process)")
            self.wait_for_user()
            return
        
        # Testa busca vetorial
        print(f"\n🔍 Testing vector search...")
        test_query = input("Enter a test query (or press Enter for default): ").strip()
        if not test_query:
            test_query = "teste"
        
        print(f"  Searching for: '{test_query}'")
        
        # Testa com threshold 0.0 primeiro
        print(f"  Testing with threshold=0.0...")
        test_embedding = self.system.embeddings.generate_embedding(test_query)
        results = self.system.db.search_similar(test_embedding, top_k=3, threshold=0.0)
        
        if results:
            print(f"  ✅ Found {len(results)} results")
            print(f"\n  Top 3 results:")
            for i, result in enumerate(results, 1):
                print(f"\n  [{i}] {result['chunk_id']}")
                print(f"      Document: {result['document_name']}")
                print(f"      Similarity: {result['similarity_score']:.4f}")
                print(f"      Preview: {result['text_content'][:80]}...")
        else:
            print(f"  ❌ No results found even with threshold=0.0")
            print(f"  This indicates a problem with embeddings or search")
            
            # Debug adicional
            print(f"\n  🔧 Checking embeddings in database...")
            conn = self.system.db.create_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(embedding) as with_embedding,
                    COUNT(*) - COUNT(embedding) as without_embedding
                FROM document_chunks
            """)
            row = cursor.fetchone()
            print(f"     Total chunks: {row[0]}")
            print(f"     With embeddings: {row[1]}")
            print(f"     Without embeddings: {row[2]}")
            
            if row[2] > 0:
                print(f"\n  ⚠️ WARNING: {row[2]} chunks are missing embeddings!")
                print(f"     Solution: Re-run option 2 (Process and index)")
            
            conn.close()
        
        print("\n" + "=" * 70)
        self.wait_for_user()
    
    def _action_clean_system(self):
        """Limpa todo o sistema"""
        print("\n" + "=" * 70)
        print("🧹 Clean System")
        print("=" * 70)
        
        confirmation = input("\n⚠️  Are you sure? All data will be lost (y/N): ").strip().lower()
        
        if confirmation == 'y':
            self.system.clean_system()
            print("✅ System cleaned successfully!")
        else:
            print("❌ Operation cancelled.")
        
        self.wait_for_user()
    
    def run(self):
        """Executa a interface CLI"""
        try:
            self.clear_screen()
            print("=" * 70)
            print("🚀 Advanced Local RAG System (Ollama + PostgreSQL)")
            print("=" * 70)
            
            # Inicializa sistema
            self._initialize_system()
            
            while True:
                self.clear_screen()
                self.show_menu()
                
                option = input("\n👉 Choose an option (0-8): ").strip()
                
                if option == "0":
                    print("\n👋 Shutting down RAG System...")
                    break
                
                # Executa acao
                action = self.actions.get(option)
                if action:
                    self.clear_screen()
                    action()
                else:
                    print("\n❌ Invalid option. Choose between 0 and 8.")
                    self.wait_for_user()
        
        except KeyboardInterrupt:
            print("\n\n⚠️  System interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Critical error: {e}")
            traceback.print_exc()
            self.wait_for_user()

def main():
    """Ponto de entrada principal"""
    cli = RagCLI()
    cli.run()

if __name__ == "__main__":
    main()