import ollama
from typing import Optional, Generator
from config.settings import LLM_MODEL, OLLAMA_TEMPERATURE, OLLAMA_TOP_P

class LLMManager:    
    def __init__(self):
        self.model = LLM_MODEL
        self.temperature = OLLAMA_TEMPERATURE
        self.top_p = OLLAMA_TOP_P
        self.client = ollama.Client(host='http://localhost:11434')
        self._check_model()
    
    def _check_model(self):
        """Verifica se o modelo está disponível"""
        try:
            models_response = self.client.list()
            
            # Extrair nomes dos modelos usando o atributo 'model'
            available_models = [model.model for model in models_response.models]
            
            if self.model in available_models:
                print(f"✅ Model {self.model} available!")
            else:
                print(f"⚠️ Model {self.model} not found in available models.")
                print(f"📋 Available models: {available_models}")
                
        except Exception as e:
            print(f"❌ Error checking models: {e}")
    
    def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """Gera resposta usando Ollama"""
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': temperature or self.temperature,
                    'top_p': top_p or self.top_p
                }
            )
            
            # Acessar a resposta usando o atributo 'response'
            return response.response
                
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error generating response: {e}"
    
    def generate_response_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """Gera transmissao de resposta"""
        try:
            stream = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                options={
                    'temperature': temperature or self.temperature
                }
            )
            
            for chunk in stream:
                yield chunk.response
        
        except Exception as e:
            print(f"❌ Error in streaming: {e}")
            yield f"Error: {e}"
    
    def get_model_info(self) -> dict:
        """Retorna informacoes do modelo"""
        try:
            models_response = self.client.list()
            
            for model in models_response.models:
                if model.model == self.model:
                    return {
                        'model': self.model,
                        'size': model.size,
                        'modified_at': model.modified_at,
                        'digest': model.digest,
                        'details': model.details
                    }
            
            return {'model': self.model, 'error': 'Model not found'}
            
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
            return {'model': self.model, 'error': str(e)}
    
    def validate_prompt(self, prompt: str) -> bool:
        """Valida prompt"""
        if not prompt or not prompt.strip():
            print("⚠️  Empty prompt provided")
            return False
        return True
    
    def generate_with_context(
        self,
        context: str,
        question: str,
        temperature: Optional[float] = None
    ) -> str:
        """Gera respostas com contexto"""
        if not self.validate_prompt(question):
            return "Please provide a valid question!"
        
        prompt = f"""
        Based on the following context, please answer the question.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        
        return self.generate_response(prompt, temperature)