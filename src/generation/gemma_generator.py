"""
Gemma 3 270M Generator for RAG Answer Generation

This module uses Google's Gemma 3 270M model to generate answers
from retrieved context after BERT routing.

Author: Sushanth Reddy
Date: October 2025
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GemmaGenerator:
    """
    Answer generator using Google Gemma 3 270M model.
    
    This class handles:
    1. Loading and initializing Gemma 3 270M
    2. Context formatting from retrieved documents
    3. Prompt construction for answer generation
    4. Response generation with configurable parameters
    """
    
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
        """
        Initialize Gemma Generator.
        
        Args:
            model_id: HuggingFace model ID for Gemma
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            max_context_length: Maximum length of context to include
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (vs greedy decoding)
        """
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
            
        logger.info(f"Initializing Gemma Generator on device: {self.device}")
        
        # Load tokenizer and model
        self._load_model()
        
    def _load_model(self):
        """Load Gemma tokenizer and model."""
        try:
            logger.info(f"Loading tokenizer from {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            logger.info(f"Loading model from {self.model_id}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            logger.info("✓ Gemma model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Gemma model: {e}")
            raise
    
    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            retrieved_chunks: List of dicts with 'content' and 'metadata' keys
            
        Returns:
            Formatted context string
        """
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
        """
        Create prompt for Gemma using retrieved context.
        
        Args:
            query: User's question
            context: Formatted context from retrieval
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a helpful customer service assistant.

INSTRUCTIONS:
- Read the context information carefully
- Answer the customer's question directly and CONCISELY IN 2-3 SENTENCES
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
        """
        Generate answer from query and retrieved context.
        
        Args:
            query: User's question
            retrieved_chunks: List of retrieved document chunks
            max_new_tokens: Override default max_new_tokens
            temperature: Override default temperature
            top_p: Override default top_p
            
        Returns:
            Dict with 'answer', 'context', and 'metadata'
        """
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
            
            # If answer is empty, provide a fallback
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
                '_generation_timing': timing_detail  # Include detailed timing
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
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
                '_generation_timing': timing_detail
            }
    
    def generate_streaming(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ):
        """
        Generate answer with streaming (for future use with UIs).
        
        Args:
            query: User's question
            retrieved_chunks: List of retrieved document chunks
            
        Yields:
            Generated tokens as they are produced
        """
        context = self.format_context(retrieved_chunks)
        prompt = self.create_prompt(query, context)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # For streaming, we'd use model.generate with streamer
        # This is a simplified version
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output[len(prompt):].strip()
        
        yield answer
    
    def __repr__(self) -> str:
        return (
            f"GemmaGenerator(model={self.model_id}, "
            f"device={self.device}, "
            f"max_new_tokens={self.max_new_tokens})"
        )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Gemma Generator...")
    generator = GemmaGenerator()
    
    # Test with sample context
    sample_chunks = [
        {
            'content': 'Brooks Ghost 15 is a neutral running shoe with DNA LOFT cushioning.',
            'metadata': {'source': 'brooks_faqs.txt'}
        }
    ]
    
    result = generator.generate(
        query="Tell me about Brooks Ghost shoes",
        retrieved_chunks=sample_chunks
    )
    
    print("\n" + "="*80)
    print("GENERATED ANSWER:")
    print("="*80)
    print(result['answer'])
    print("\n" + "="*80)
