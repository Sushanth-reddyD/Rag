"""
Test Gemma Generation Speed with Different Input Lengths

This script tests how Gemma's generation speed varies with different
input context lengths. It measures:
- Tokens per second
- Total inference time
- Context formatting time
- Tokenization time

Author: Sushanth Reddy
Date: October 2025
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.generation.model_factory import ModelFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_context(length_chars: int) -> str:
    """
    Create test context of specified length.
    
    Args:
        length_chars: Approximate number of characters
        
    Returns:
        Test context string
    """
    base_text = """A leading manufacturer of high-performance running shoes, 
apparel, and accessories. Founded in 1914, Brooks has been dedicated to inspiring people to run 
and be active. Our products are designed with biomechanical research and advanced materials to 
provide runners with the best possible experience. We offer a wide range of shoes for different 
running styles including neutral, stability, and motion control. Popular models include the Ghost, 
Adrenaline GTS, Glycerin, and Levitate series. Each shoe is engineered with technologies like 
DNA LOFT cushioning, GuideRails support system, and BioMoGo DNA for responsive comfort. """
    
    # Repeat text to reach desired length
    repetitions = (length_chars // len(base_text)) + 1
    full_text = (base_text * repetitions)[:length_chars]
    
    return full_text


def create_test_chunks(context: str, source_name: str = "test_document.txt") -> List[Dict[str, Any]]:
    """
    Create test chunks from context.
    
    Args:
        context: Context text
        source_name: Source document name
        
    Returns:
        List of chunk dictionaries
    """
    return [{
        'content': context,
        'metadata': {
            'source': source_name,
            'chunk_id': 0
        }
    }]


def test_generation_speed(
    generator,
    query: str,
    context_lengths: List[int],
    max_new_tokens: int = 100
):
    """
    Test generation speed with different context lengths.
    
    Args:
        generator: Generator instance
        query: Test query
        context_lengths: List of context lengths to test (in characters)
        max_new_tokens: Maximum tokens to generate
    """
    print("\n" + "="*100)
    print("🚀 GEMMA GENERATION SPEED TEST")
    print("="*100)
    print(f"\nTest Query: '{query}'")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Context Lengths to Test: {context_lengths}")
    print(f"Model: {generator.model_id}")
    print(f"Device: {generator.device}")
    print("\n" + "="*100 + "\n")
    
    results = []
    
    for length in context_lengths:
        print(f"\n{'─'*100}")
        print(f"📏 TESTING CONTEXT LENGTH: {length:,} characters (~{length//4} tokens)")
        print(f"{'─'*100}\n")
        
        # Create test context
        context = create_test_context(length)
        chunks = create_test_chunks(context)
        
        # Warm-up run (not counted)
        if length == context_lengths[0]:
            print("🔥 Warming up model (first run, not counted)...")
            _ = generator.generate(
                query=query,
                retrieved_chunks=chunks,
                max_new_tokens=max_new_tokens
            )
            print("✓ Warm-up complete\n")
            time.sleep(1)
        
        # Actual test run
        print(f"⏱️  Starting timed generation...")
        start_time = time.time()
        
        result = generator.generate(
            query=query,
            retrieved_chunks=chunks,
            max_new_tokens=max_new_tokens
        )
        
        total_time = time.time() - start_time
        
        # Extract timing details
        timing = result.get('_generation_timing', {})
        
        # Calculate metrics
        tokens_per_sec = timing.get('tokens_per_second', 0)
        input_tokens = timing.get('input_tokens', 0)
        output_tokens = timing.get('output_tokens', 0)
        inference_ms = timing.get('model_inference_ms', 0)
        
        # Store results
        results.append({
            'context_length_chars': length,
            'context_length_tokens_approx': length // 4,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_time_ms': total_time * 1000,
            'inference_time_ms': inference_ms,
            'tokens_per_second': tokens_per_sec,
            'format_context_ms': timing.get('format_context_ms', 0),
            'tokenization_ms': timing.get('tokenization_ms', 0),
            'decoding_ms': timing.get('decoding_ms', 0),
            'answer_preview': result['answer'][:100] + '...' if len(result['answer']) > 100 else result['answer']
        })
        
        print(f"\n✅ Test completed for {length:,} chars")
        print(f"   └─ Total time: {total_time*1000:.2f} ms\n")
        
        # Small delay between tests
        time.sleep(2)
    
    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print summary of test results."""
    print("\n" + "="*100)
    print("📊 GENERATION SPEED TEST RESULTS SUMMARY")
    print("="*100 + "\n")
    
    # Print table header
    print(f"{'Context':<12} {'Input':<8} {'Output':<8} {'Inference':<12} {'Total':<12} {'Speed':<12} {'Efficiency'}")
    print(f"{'Length':<12} {'Tokens':<8} {'Tokens':<8} {'Time (ms)':<12} {'Time (ms)':<12} {'(tok/s)':<12} {'%'}")
    print("─" * 100)
    
    for r in results:
        context_str = f"{r['context_length_chars']:,} ch"
        input_tok = f"{r['input_tokens']}"
        output_tok = f"{r['output_tokens']}"
        inference_ms = f"{r['inference_time_ms']:.0f}"
        total_ms = f"{r['total_time_ms']:.0f}"
        speed = f"{r['tokens_per_second']:.1f}"
        efficiency = f"{(r['inference_time_ms'] / r['total_time_ms'] * 100):.1f}%"
        
        print(f"{context_str:<12} {input_tok:<8} {output_tok:<8} {inference_ms:<12} {total_ms:<12} {speed:<12} {efficiency}")
    
    print("\n" + "="*100 + "\n")
    
    # Analysis
    print("📈 ANALYSIS:\n")
    
    # Speed trend
    speeds = [r['tokens_per_second'] for r in results]
    avg_speed = sum(speeds) / len(speeds)
    min_speed = min(speeds)
    max_speed = max(speeds)
    
    print(f"   Speed Statistics:")
    print(f"   ├─ Average: {avg_speed:.2f} tokens/sec")
    print(f"   ├─ Minimum: {min_speed:.2f} tokens/sec (at {results[speeds.index(min_speed)]['context_length_chars']:,} chars)")
    print(f"   └─ Maximum: {max_speed:.2f} tokens/sec (at {results[speeds.index(max_speed)]['context_length_chars']:,} chars)")
    
    # Context length impact
    print(f"\n   Context Length Impact:")
    if len(results) > 1:
        speed_diff = speeds[-1] - speeds[0]
        speed_diff_pct = (speed_diff / speeds[0]) * 100
        
        if abs(speed_diff_pct) < 10:
            print(f"   ├─ Minimal impact: {speed_diff_pct:+.1f}% change from shortest to longest")
            print(f"   └─ ✅ Speed is relatively consistent across context lengths")
        elif speed_diff_pct < 0:
            print(f"   ├─ Negative impact: {speed_diff_pct:.1f}% slower with longer context")
            print(f"   └─ ⚠️  Longer contexts reduce generation speed")
        else:
            print(f"   ├─ Positive impact: {speed_diff_pct:+.1f}% faster with longer context")
            print(f"   └─ ✅ Performance improves with longer contexts (warm-up effect)")
    
    # Timing breakdown (last test)
    print(f"\n   Timing Breakdown (longest context):")
    last = results[-1]
    print(f"   ├─ Context Formatting: {last['format_context_ms']:.2f} ms")
    print(f"   ├─ Tokenization: {last['tokenization_ms']:.2f} ms")
    print(f"   ├─ Model Inference: {last['inference_time_ms']:.2f} ms ({last['inference_time_ms']/last['total_time_ms']*100:.1f}% of total)")
    print(f"   └─ Decoding: {last['decoding_ms']:.2f} ms")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:\n")
    
    if avg_speed < 10:
        print(f"   ⚠️  Very slow generation speed ({avg_speed:.1f} tok/s)")
        print(f"   Suggestions:")
        print(f"   1. Use GPU if available (currently using: {results[0].get('device', 'unknown')})")
        print(f"   2. Switch to Gemini API for faster response")
        print(f"   3. Reduce max_new_tokens for shorter answers")
        print(f"   4. Consider model quantization (int8/int4)")
    elif avg_speed < 20:
        print(f"   ⚠️  Moderate generation speed ({avg_speed:.1f} tok/s)")
        print(f"   Suggestions:")
        print(f"   1. GPU acceleration would significantly improve speed")
        print(f"   2. Acceptable for low-traffic applications")
    else:
        print(f"   ✅ Good generation speed ({avg_speed:.1f} tok/s)")
        print(f"   Current performance is acceptable for production use")
    
    # Input token impact
    input_tokens = [r['input_tokens'] for r in results]
    if len(input_tokens) > 1:
        token_increase = input_tokens[-1] - input_tokens[0]
        time_increase = results[-1]['inference_time_ms'] - results[0]['inference_time_ms']
        ms_per_token = time_increase / token_increase if token_increase > 0 else 0
        
        print(f"\n   Input Token Impact:")
        print(f"   └─ ~{ms_per_token:.2f} ms additional inference time per input token")
    
    print("\n" + "="*100 + "\n")


def main():
    """Main test function."""
    print("\n🔧 Initializing Gemma model...")
    print("   (This may take a minute on first load)\n")
    
    try:
        # Create Gemma generator
        generator = ModelFactory.create_generator(
            model_type='gemma',
            use_config_file=True
        )
        
        print(f"✓ Model loaded: {generator.model_id}")
        print(f"✓ Device: {generator.device}\n")
        
        # Test configuration
        test_query = "What is the life expectancy of running shoes?"
        
        # Test different context lengths (characters)
        context_lengths = [
            500,      # ~125 tokens - Very short
            2000,     # ~500 tokens - Short
            4000,     # ~1000 tokens - Medium
            8000,     # ~2000 tokens - Long
            16000,    # ~4000 tokens - Very long
        ]
        
        # Maximum tokens to generate (keep consistent)
        max_new_tokens = 100
        
        # Run tests
        results = test_generation_speed(
            generator=generator,
            query=test_query,
            context_lengths=context_lengths,
            max_new_tokens=max_new_tokens
        )
        
        # Print summary
        print_summary(results)
        
        # Sample answer
        print("\n📝 SAMPLE ANSWER (from longest context):")
        print("─" * 100)
        print(results[-1]['answer_preview'])
        print("─" * 100 + "\n")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
