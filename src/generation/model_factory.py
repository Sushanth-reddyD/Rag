"""
Model Factory for Dynamic Model Selection

This module provides a factory pattern for selecting and initializing
different generation models (Gemma, Gemini) based on environment variables.

Author: Sushanth Reddy
Date: October 2025
"""

import os
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    GEMMA = "gemma"
    GEMINI = "gemini"


class BaseGenerator:
    """Base class for all generators."""
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate answer from query and retrieved context."""
        raise NotImplementedError("Subclasses must implement generate()")
    
    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string."""
        raise NotImplementedError("Subclasses must implement format_context()")


class GemmaGenerator(BaseGenerator):
    """Gemma 3 270M Generator (Local Inference)."""
    
    def __init__(
        self,
        model_id: str = "google/gemma-3-270m-it",
        device: Optional[str] = None,
        max_context_length: int = 1024,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ):
        """Initialize Gemma generator."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "Gemma requires transformers and torch. "
                "Install with: pip install transformers torch accelerate"
            ) from e
        
        self.model_id = model_id
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading Gemma model: {self.model_id} on {self.device}")
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("✓ Gemma model loaded successfully")
    
    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string."""
        if not retrieved_chunks:
            return "No relevant information found."
        
        context_parts = []
        for chunk in retrieved_chunks:
            content = chunk.get('content', '')
            # Just add the content without source labels
            # The sources will be displayed separately in the UI
            if content.strip():
                context_parts.append(content.strip())
        
        full_context = "\n\n".join(context_parts)
        
        # Truncate if too long
        tokens = self.tokenizer.encode(full_context, add_special_tokens=False)
        if len(tokens) > self.max_context_length:
            tokens = tokens[:self.max_context_length]
            full_context = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return full_context
    
    def create_prompt(self, query: str, context: str) -> str:
        """Create prompt for Gemma."""
        prompt = f"""You are a helpful customer service assistant.

INSTRUCTIONS:
- Read the context information carefully
- Answer the customer's question directly and concisely in 2-3 sentences
- Use only information from the context provided
- Write in a natural, conversational tone
- If the context doesn't contain relevant information, say "I don't have information about that in our knowledge base."

OUTPUT FORMAT - You MUST respond with ONLY a valid JSON object in this exact format:
{{
    "answer": "Your 2-3 sentence answer here"
}}

DO NOT include any text before or after the JSON. DO NOT include source labels, metadata, or the question.

CONTEXT INFORMATION:
{context}

CUSTOMER QUESTION:
{query}

JSON RESPONSE:"""
        return prompt
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate answer using Gemma."""
        import torch
        import json
        import re
        import time
        
        # Detailed timing breakdown
        timing_detail = {}
        total_start = time.time()
        
        print("\n" + "="*80)
        print("🤖 GEMMA GENERATION - DETAILED TIMING BREAKDOWN")
        print("="*80)
        
        try:
            # 1. Format context
            step_start = time.time()
            context = self.format_context(retrieved_chunks)
            timing_detail['format_context_ms'] = (time.time() - step_start) * 1000
            print(f"📝 Context Formatting:     {timing_detail['format_context_ms']:.2f} ms")
            print(f"   └─ Context length: {len(context)} chars")
            
            # 2. Create prompt
            step_start = time.time()
            prompt = self.create_prompt(query, context)
            timing_detail['create_prompt_ms'] = (time.time() - step_start) * 1000
            print(f"✏️  Prompt Creation:        {timing_detail['create_prompt_ms']:.2f} ms")
            print(f"   └─ Prompt length: {len(prompt)} chars")
            
            # 3. Tokenize
            step_start = time.time()
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            timing_detail['tokenization_ms'] = (time.time() - step_start) * 1000
            timing_detail['input_tokens'] = inputs['input_ids'].shape[1]
            print(f"🔤 Tokenization:           {timing_detail['tokenization_ms']:.2f} ms")
            print(f"   └─ Input tokens: {timing_detail['input_tokens']}")
            
            # 4. Model inference (generation)
            step_start = time.time()
            max_tokens = max_new_tokens or self.max_new_tokens
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature or self.temperature,
                    top_p=top_p or self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            timing_detail['model_inference_ms'] = (time.time() - step_start) * 1000
            timing_detail['output_tokens'] = outputs.shape[1] - timing_detail['input_tokens']
            timing_detail['tokens_per_second'] = timing_detail['output_tokens'] / (timing_detail['model_inference_ms'] / 1000)
            
            print(f"🧠 Model Inference:        {timing_detail['model_inference_ms']:.2f} ms ⚠️ SLOWEST")
            print(f"   ├─ Generated tokens: {timing_detail['output_tokens']}")
            print(f"   ├─ Max tokens allowed: {max_tokens}")
            print(f"   ├─ Tokens/second: {timing_detail['tokens_per_second']:.2f}")
            print(f"   ├─ Device: {self.device}")
            print(f"   └─ Model: {self.model_id}")
            
            # 5. Decode output
            step_start = time.time()
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            timing_detail['decoding_ms'] = (time.time() - step_start) * 1000
            print(f"📤 Decoding Output:        {timing_detail['decoding_ms']:.2f} ms")
            
            # 6. Extract answer
            step_start = time.time()
            raw_answer = full_output[len(prompt):].strip()
            timing_detail['extract_answer_ms'] = (time.time() - step_start) * 1000
            print(f"✂️  Extract Answer:         {timing_detail['extract_answer_ms']:.2f} ms")
            
            # 7. Parse JSON response
            step_start = time.time()
            parsed_answer = raw_answer
            try:
                # Try to find and parse JSON object
                json_match = re.search(r'\{[\s\S]*?"answer"[\s\S]*?:[\s\S]*?"([^"]*)"[\s\S]*?\}', raw_answer)
                if json_match:
                    # Extract the answer value from the regex match
                    parsed_answer = json_match.group(1)
                else:
                    # Try direct JSON parse
                    json_obj = json.loads(raw_answer)
                    parsed_answer = json_obj.get('answer', raw_answer)
            except (json.JSONDecodeError, AttributeError):
                # If JSON parsing fails, clean up the raw answer
                parsed_answer = re.sub(r'^\{.*?"answer".*?:\s*"', '', raw_answer)
                parsed_answer = re.sub(r'"\s*\}\s*$', '', parsed_answer)
                parsed_answer = raw_answer if not parsed_answer.strip() else parsed_answer
            
            timing_detail['json_parsing_ms'] = (time.time() - step_start) * 1000
            print(f"🔍 JSON Parsing:           {timing_detail['json_parsing_ms']:.2f} ms")
            
            if not parsed_answer or not parsed_answer.strip():
                parsed_answer = "I apologize, but I couldn't generate a specific answer based on the available context."
            
            # Calculate total time
            timing_detail['total_generation_ms'] = (time.time() - total_start) * 1000
            
            print(f"\n{'─'*80}")
            print(f"⏱️  TOTAL GENERATION TIME:  {timing_detail['total_generation_ms']:.2f} ms")
            print(f"   └─ Model inference is {(timing_detail['model_inference_ms']/timing_detail['total_generation_ms']*100):.1f}% of total")
            print("="*80 + "\n")
            
            # Performance analysis
            if timing_detail['tokens_per_second'] < 10:
                print("⚠️  WARNING: Very slow generation (<10 tokens/sec)")
                print("💡 Recommendations:")
                if self.device == 'cpu':
                    print("   1. Use GPU: Set device='cuda' in model_config.py")
                    print("   2. Or switch to Gemini API for faster response")
                print("   3. Reduce max_new_tokens to generate shorter answers")
            elif timing_detail['tokens_per_second'] < 20:
                print("⚠️  NOTICE: Moderate generation speed (10-20 tokens/sec)")
                print("💡 Consider GPU acceleration for better performance")
            
            return {
                'answer': parsed_answer,
                'context': context,
                'query': query,
                'num_sources': len(retrieved_chunks),
                'sources': [chunk.get('metadata', {}).get('source', 'Unknown') 
                           for chunk in retrieved_chunks],
                'model': self.model_id,
                'model_type': 'gemma',
                '_generation_timing': timing_detail  # Include detailed timing
            }
        except Exception as e:
            logger.error(f"Error generating with Gemma: {e}")
            timing_detail['total_generation_ms'] = (time.time() - total_start) * 1000
            timing_detail['error'] = str(e)
            
            print(f"❌ Generation Error: {e}")
            print(f"⏱️  Time before error: {timing_detail['total_generation_ms']:.2f} ms")
            print("="*80 + "\n")
            
            return {
                'answer': f"Error generating answer: {str(e)}",
                'context': context if 'context' in locals() else '',
                'query': query,
                'error': str(e),
                'model_type': 'gemma',
                '_generation_timing': timing_detail
            }


class GeminiGenerator(BaseGenerator):
    """Gemini 2.5 Pro/Flash Generator (API-based)."""
    
    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_output_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Initialize Gemini generator.
        
        Args:
            model_id: Gemini model ID (gemini-2.5-flash, gemini-2.5-pro, etc.)
            api_key: Gemini API key (if None, uses GEMINI_API_KEY env variable)
            max_output_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "Gemini requires google-genai package. "
                "Install with: pip install google-genai"
            ) from e
        
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Get API key from env or parameter
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        logger.info(f"Initializing Gemini model: {self.model_id}")
        
        # Initialize client
        self.client = genai.Client(api_key=self.api_key)
        
        logger.info("✓ Gemini client initialized successfully")
    
    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string."""
        if not retrieved_chunks:
            return "No relevant information found."
        
        context_parts = []
        for chunk in retrieved_chunks:
            content = chunk.get('content', '')
            # Just add the content without source labels
            # The sources will be displayed separately in the UI
            if content.strip():
                context_parts.append(content.strip())
        
        return "\n\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """Create prompt for Gemini."""
        prompt = f"""You are a helpful customer service assistant.

INSTRUCTIONS:
- Read the context information carefully
- Answer the customer's question directly and concisely in 2-3 sentences
- Use only information from the context provided
- Write in a natural, conversational tone
- If the context doesn't contain relevant information, say "I don't have information about that in our knowledge base."

OUTPUT FORMAT - You MUST respond with ONLY a valid JSON object in this exact format:
{{
    "answer": "Your 2-3 sentence answer here"
}}

DO NOT include any text before or after the JSON. DO NOT include source labels, metadata, or the question.

CONTEXT INFORMATION:
{context}

CUSTOMER QUESTION:
{query}

JSON RESPONSE:"""
        return prompt
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate answer using Gemini API."""
        import json
        import re
        import time
        
        # Detailed timing breakdown
        timing_detail = {}
        total_start = time.time()
        
        print("\n" + "="*80)
        print("🤖 GEMINI GENERATION - DETAILED TIMING BREAKDOWN")
        print("="*80)
        
        try:
            # 1. Format context
            step_start = time.time()
            context = self.format_context(retrieved_chunks)
            timing_detail['format_context_ms'] = (time.time() - step_start) * 1000
            print(f"📝 Context Formatting:     {timing_detail['format_context_ms']:.2f} ms")
            print(f"   └─ Context length: {len(context)} chars")
            
            # 2. Create prompt
            step_start = time.time()
            prompt = self.create_prompt(query, context)
            timing_detail['create_prompt_ms'] = (time.time() - step_start) * 1000
            print(f"✏️  Prompt Creation:        {timing_detail['create_prompt_ms']:.2f} ms")
            print(f"   └─ Prompt length: {len(prompt)} chars")
            
            # 3. Configure generation parameters
            step_start = time.time()
            config = {
                'temperature': temperature or self.temperature,
                'top_p': top_p or self.top_p,
                'max_output_tokens': max_output_tokens or self.max_output_tokens,
            }
            timing_detail['config_setup_ms'] = (time.time() - step_start) * 1000
            print(f"⚙️  Config Setup:           {timing_detail['config_setup_ms']:.2f} ms")
            
            # 4. API Call (generation)
            step_start = time.time()
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=config
            )
            timing_detail['api_call_ms'] = (time.time() - step_start) * 1000
            print(f"🌐 Gemini API Call:        {timing_detail['api_call_ms']:.2f} ms ⚠️ SLOWEST")
            print(f"   ├─ Model: {self.model_id}")
            print(f"   ├─ Max output tokens: {config['max_output_tokens']}")
            print(f"   └─ Network + inference time")
            
            # 5. Extract response
            step_start = time.time()
            raw_answer = response.text.strip()
            timing_detail['extract_response_ms'] = (time.time() - step_start) * 1000
            timing_detail['response_length'] = len(raw_answer)
            print(f"📤 Extract Response:       {timing_detail['extract_response_ms']:.2f} ms")
            print(f"   └─ Response length: {timing_detail['response_length']} chars")
            
            # 6. Parse JSON response
            step_start = time.time()
            parsed_answer = raw_answer
            try:
                # Try to find and parse JSON object
                json_match = re.search(r'\{[\s\S]*?"answer"[\s\S]*?:[\s\S]*?"([^"]*)"[\s\S]*?\}', raw_answer)
                if json_match:
                    # Extract the answer value from the regex match
                    parsed_answer = json_match.group(1)
                else:
                    # Try direct JSON parse
                    json_obj = json.loads(raw_answer)
                    parsed_answer = json_obj.get('answer', raw_answer)
            except (json.JSONDecodeError, AttributeError):
                # If JSON parsing fails, clean up the raw answer
                parsed_answer = re.sub(r'^\{.*?"answer".*?:\s*"', '', raw_answer)
                parsed_answer = re.sub(r'"\s*\}\s*$', '', parsed_answer)
                parsed_answer = raw_answer if not parsed_answer.strip() else parsed_answer
            
            timing_detail['json_parsing_ms'] = (time.time() - step_start) * 1000
            print(f"🔍 JSON Parsing:           {timing_detail['json_parsing_ms']:.2f} ms")
            
            if not parsed_answer or not parsed_answer.strip():
                parsed_answer = "I apologize, but I couldn't generate a specific answer based on the available context."
            
            # Calculate total time
            timing_detail['total_generation_ms'] = (time.time() - total_start) * 1000
            
            print(f"\n{'─'*80}")
            print(f"⏱️  TOTAL GENERATION TIME:  {timing_detail['total_generation_ms']:.2f} ms")
            print(f"   └─ API call is {(timing_detail['api_call_ms']/timing_detail['total_generation_ms']*100):.1f}% of total")
            print("="*80 + "\n")
            
            # Performance analysis
            if timing_detail['api_call_ms'] > 2000:
                print("⚠️  WARNING: Slow API response (>2 seconds)")
                print("💡 Possible causes:")
                print("   1. Network latency")
                print("   2. High API load")
                print("   3. Consider using gemini-2.5-flash for faster responses")
            elif timing_detail['api_call_ms'] > 1000:
                print("⚠️  NOTICE: Moderate API latency (1-2 seconds)")
            
            return {
                'answer': parsed_answer,
                'context': context,
                'query': query,
                'num_sources': len(retrieved_chunks),
                'sources': [chunk.get('metadata', {}).get('source', 'Unknown') 
                           for chunk in retrieved_chunks],
                'model': self.model_id,
                'model_type': 'gemini',
                '_generation_timing': timing_detail  # Include detailed timing
            }
        except Exception as e:
            logger.error(f"Error generating with Gemini: {e}")
            timing_detail['total_generation_ms'] = (time.time() - total_start) * 1000
            timing_detail['error'] = str(e)
            
            print(f"❌ Generation Error: {e}")
            print(f"⏱️  Time before error: {timing_detail['total_generation_ms']:.2f} ms")
            print("="*80 + "\n")
            
            return {
                'answer': f"Error generating answer: {str(e)}",
                'context': context if 'context' in locals() else '',
                'query': query,
                'error': str(e),
                'model_type': 'gemini',
                '_generation_timing': timing_detail
            }


class ModelFactory:
    """
    Factory class for creating generation models based on configuration.
    
    Supports:
    - Gemma 3 270M (local inference)
    - Gemini 2.5 Pro/Flash (API-based)
    """
    
    @staticmethod
    def create_generator(
        model_type: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        use_config_file: bool = True
    ) -> BaseGenerator:
        """
        Create a generator based on model type.
        
        Args:
            model_type: Type of model ('gemma', 'gemini', or None for config-based)
            model_config: Configuration dictionary for the model
            use_config_file: If True, read from src/config/model_config.py (default)
            
        Returns:
            Generator instance (GemmaGenerator or GeminiGenerator)
        
        Configuration:
            Set MODEL_TYPE and GEMINI_API_KEY in src/config/model_config.py
        """
        if model_config is None:
            model_config = {}
        
        # Load from config file if requested
        if use_config_file and model_type is None:
            try:
                from ..config import model_config as config
                model_type = config.MODEL_TYPE
                logger.info(f"Loaded model type from config: {model_type}")
            except ImportError:
                logger.warning("Could not load config file, using defaults")
                model_type = 'gemma'
        
        # Determine model type from parameter or environment
        if model_type is None:
            model_type = os.getenv('MODEL_TYPE', 'gemma').lower()
        else:
            model_type = model_type.lower()
        
        logger.info(f"Creating generator: {model_type}")
        
        # Create appropriate generator
        if model_type == 'gemma':
            return ModelFactory._create_gemma_generator(model_config, use_config_file)
        elif model_type == 'gemini':
            return ModelFactory._create_gemini_generator(model_config, use_config_file)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported types: 'gemma', 'gemini'"
            )
    
    @staticmethod
    def _create_gemma_generator(config: Dict[str, Any], use_config_file: bool = True) -> GemmaGenerator:
        """Create Gemma generator with config."""
        # Try to load from config file first
        if use_config_file:
            try:
                from ..config import model_config as cfg
                model_id = config.get('model_id') or cfg.MODEL_ID or 'google/gemma-3-270m-it'
                gemma_cfg = cfg.GEMMA_CONFIG
                logger.info(f"Using Gemma config from model_config.py")
            except ImportError:
                logger.warning("Could not load model_config.py, using defaults")
                model_id = config.get('model_id') or os.getenv('MODEL_ID', 'google/gemma-3-270m-it')
                gemma_cfg = {}
        else:
            model_id = config.get('model_id') or os.getenv('MODEL_ID', 'google/gemma-3-270m-it')
            gemma_cfg = {}
        
        return GemmaGenerator(
            model_id=model_id,
            device=config.get('device') or gemma_cfg.get('device'),
            max_context_length=config.get('max_context_length') or gemma_cfg.get('max_context_length', 1024),
            max_new_tokens=config.get('max_new_tokens') or gemma_cfg.get('max_new_tokens', 256),
            temperature=config.get('temperature') or gemma_cfg.get('temperature', 0.7),
            top_p=config.get('top_p') or gemma_cfg.get('top_p', 0.9),
            do_sample=config.get('do_sample', gemma_cfg.get('do_sample', True))
        )
    
    @staticmethod
    def _create_gemini_generator(config: Dict[str, Any], use_config_file: bool = True) -> GeminiGenerator:
        """Create Gemini generator with config."""
        # Try to load from config file first
        if use_config_file:
            try:
                from ..config import model_config as cfg
                model_id = config.get('model_id') or cfg.MODEL_ID or 'gemini-2.5-flash'
                api_key = config.get('api_key') or cfg.GEMINI_API_KEY
                gemini_cfg = cfg.GEMINI_CONFIG
                logger.info(f"Using Gemini config from model_config.py")
            except ImportError:
                logger.warning("Could not load model_config.py, using defaults")
                model_id = config.get('model_id') or os.getenv('MODEL_ID', 'gemini-2.5-flash')
                api_key = config.get('api_key') or os.getenv('GEMINI_API_KEY')
                gemini_cfg = {}
        else:
            model_id = config.get('model_id') or os.getenv('MODEL_ID', 'gemini-2.5-flash')
            api_key = config.get('api_key') or os.getenv('GEMINI_API_KEY')
            gemini_cfg = {}
        
        return GeminiGenerator(
            model_id=model_id,
            api_key=api_key,
            max_output_tokens=config.get('max_output_tokens') or gemini_cfg.get('max_output_tokens', 512),
            temperature=config.get('temperature') or gemini_cfg.get('temperature', 0.7),
            top_p=config.get('top_p') or gemini_cfg.get('top_p', 0.9)
        )
    
    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """Get list of available models by type."""
        return {
            'gemma': [
                'google/gemma-3-270m-it',
                'google/gemma-2b',
                'google/gemma-7b'
            ],
            'gemini': [
                'gemini-2.5-flash',
                'gemini-2.5-pro',
                'gemini-2.0-flash-exp',
                'gemini-1.5-pro',
                'gemini-1.5-flash'
            ]
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("Model Factory Examples")
    print("="*80 + "\n")
    
    # Example 1: Create Gemma generator (default)
    print("1. Creating Gemma generator...")
    try:
        gemma_gen = ModelFactory.create_generator(model_type='gemma')
        print(f"   ✓ Created: {gemma_gen.model_id}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Example 2: Create Gemini generator
    print("\n2. Creating Gemini generator...")
    try:
        gemini_gen = ModelFactory.create_generator(
            model_type='gemini',
            model_config={'model_id': 'gemini-2.5-flash'}
        )
        print(f"   ✓ Created: {gemini_gen.model_id}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Example 3: Show available models
    print("\n3. Available models:")
    models = ModelFactory.get_available_models()
    for model_type, model_list in models.items():
        print(f"\n   {model_type.upper()}:")
        for model in model_list:
            print(f"     - {model}")
    
    print("\n" + "="*80 + "\n")
