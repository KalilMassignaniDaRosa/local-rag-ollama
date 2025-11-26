import ollama
import psycopg2
from sentence_transformers import SentenceTransformer
import json
import sys
import os

def print_success(message: str):
    """Print with green color"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print with red color"""
    print(f"❌ {message}")

def print_warning(message: str):
    """Print with yellow color"""
    print(f"⚠️  {message}")

def print_info(message: str):
    """Print info message"""
    print(f"ℹ️  {message}")

def test_ollama():
    """Test Ollama LLM functionality"""
    print("\n1. 🧠 Testing Ollama...")
    try:
        # Criar cliente com host
        client = ollama.Client(host='http://localhost:11434')
        
        # Teste de disponibilidade
        models_response = client.list()
        available_models = [model.model for model in models_response.models]
        print_info(f"Available models: {', '.join(available_models)}")
        
        # Teste para geracao com um prompt simples
        response = client.generate(model='llama3.2:3b', prompt='Ola!')
        answer = response.response.strip()
        print_success(f"Ollama response: {answer}")
        return True
        
    except Exception as e:
        print_error(f"Ollama error: {e}")
        print_warning("Make sure Ollama is running: 'ollama serve'")
        return False

def test_postgresql():
    """Test PostgreSQL and pgvector functionality"""
    print("\n2. 🗄️ Testing PostgreSQL...")
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5433",
            database="rag_database",
            user="postgres",
            password="postgres"
        )
        cursor = conn.cursor()
        
        # Testa extensao de vetor
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        vector_installed = cursor.fetchone()
        if vector_installed:
            print_success("pgVector extension: Installed")
        else:
            print_error("pgVector extension: Not installed")
            print_warning("Run: CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Testa existencia da tabela
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'document_chunks'
            );
        """)
        table_exists = cursor.fetchone()[0]
        if table_exists:
            print_success("Table document_chunks: Exists")
            
            # Testa estrutura da tabela
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'document_chunks'
            """)
            columns = cursor.fetchall()
            print_info(f"Table columns: {len(columns)}")
            
        else:
            print_warning("Table document_chunks: Does not exist")
            print_info("It will be created automatically when the system runs")
        
        # Testa operacoes basicas
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print_info(f"PostgreSQL version: {version.split(',')[0]}")
        
        conn.close()
        return vector_installed is not None
        
    except Exception as e:
        print_error(f"PostgreSQL error: {e}")
        print_warning("Make sure PostgreSQL is running on port 5433")
        return False

def test_embeddings():
    """Testa embeddings"""
    print("\n3. 🔤 Testing embedding model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_text = "This is a test sentence for embedding generation"
        embedding = model.encode(test_text)
        
        print_success(f"Sentence Transformers working! Dimension: {embedding.shape}")
        print_info(f"Model: {model._modules['0']._modules['auto_model'].config.name_or_path}")
        
        # Testa embeddings em sequencia
        batch_texts = ["First sentence", "Second sentence", "Third sentence"]
        batch_embeddings = model.encode(batch_texts)
        print_success(f"Batch processing: {len(batch_embeddings)} embeddings generated")
        
        return True
        
    except Exception as e:
        print_error(f"Embeddings error: {e}")
        return False

def test_transformers():
    """Testa Hugging Face transformers"""
    print("\n4. 🤗 Testing transformers library...")
    try:
        from transformers import pipeline
        
        # Testa analise de sentimentos
        classifier = pipeline("sentiment-analysis")
        result = classifier("I love programming in Python!")[0]
        
        print_success(f"Transformers working! Sentiment: {result['label']} (score: {result['score']:.3f})")
        
        # Test multiple models availability
        try:
            generator = pipeline("text-generation", model="distilgpt2", max_length=50)
            generated = generator("The future of AI is", num_return_sequences=1)
            print_success("Text generation pipeline working")
        except Exception as e:
            print_warning(f"Text generation test failed: {e}")
            
        return True
        
    except Exception as e:
        print_error(f"Transformers error: {e}")
        print_warning("Install with: pip install transformers torch")
        return False

def test_system_dependencies():
    """Testa Python e dependencias do sistema"""
    print("\n5. 🔧 Testing system dependencies...")
    
    dependencies = {
        'Python Version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'Ollama Python': ollama.__version__ if hasattr(ollama, '__version__') else 'Available',
        'PyPDF2': 'Available',
        'NumPy': 'Available',
        'SentenceTransformers': 'Available'
    }
    
    # Testa PyPDF2
    try:
        import PyPDF2
        dependencies['PyPDF2'] = PyPDF2.__version__
    except ImportError:
        dependencies['PyPDF2'] = 'Missing'
    
    # Testa NumPy
    try:
        import numpy as np
        dependencies['NumPy'] = np.__version__
    except ImportError:
        dependencies['NumPy'] = 'Missing'
    
    # Testa SentenceTransformers versao
    try:
        import sentence_transformers
        dependencies['SentenceTransformers'] = sentence_transformers.__version__
    except ImportError:
        dependencies['SentenceTransformers'] = 'Missing'
    
    # Imprime status de dependencia
    for dep, status in dependencies.items():
        if 'Missing' in status:
            print_error(f"{dep}: {status}")
        else:
            print_success(f"{dep}: {status}")
    
    return all('Missing' not in status for status in dependencies.values())

def main():
    print("🧪 Testing Complete RAG System...")
    print("=" * 60)
    
    # Resultados dos testes
    test_results = {}
    
    # Roda todos os testes
    test_results['ollama'] = test_ollama()
    test_results['postgresql'] = test_postgresql()
    test_results['embeddings'] = test_embeddings()
    test_results['transformers'] = test_transformers()
    test_results['dependencies'] = test_system_dependencies()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name.upper():<15}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("🎉 All tests passed! System is ready to use.")
        print("\n🚀 You can now run:")
        print("   python config\\cli_interface.py")
        print("   python rag_chat.py")
    else:
        print_warning("⚠️  Some tests failed. Check the warnings above.")
        print_info("The system may still work for basic functionality.")
    
    # Proximos passos
    print("\n🚀 NEXT STEPS:")
    if not test_results['ollama']:
        print("- Start Ollama: 'ollama serve'")
        print("- Pull model: 'ollama pull llama3.2:3b'")
        print("- Check connection: 'curl http://localhost:11434/api/tags'")
    
    if not test_results['postgresql']:
        print("- Start PostgreSQL on port 5433")
        print("- Create database: 'createdb rag_database'")
        print("- Install extension: 'CREATE EXTENSION vector;'")
    
    if not test_results['embeddings']:
        print("- Install: 'pip install sentence-transformers'")
    
    if not test_results['transformers']:
        print("- Install: 'pip install transformers torch'")

if __name__ == "__main__":
    main()