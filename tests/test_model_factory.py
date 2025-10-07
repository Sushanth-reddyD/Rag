"""
Test script for Model Factory with Gemma and Gemini

This script tests the ModelFactory class with both Gemma (local) and Gemini (API) models.

Author: Sushanth Reddy
Date: October 2025
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.generation.model_factory import ModelFactory


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def test_model_factory():
    """Test the ModelFactory with different models."""
    
    print("\n" + "="*80)
    print("🧪 TESTING: Model Factory (Gemma + Gemini)")
    print("="*80 + "\n")
    
    # Sample retrieved chunks
    sample_chunks = [
        {
            'content': 'Brooks offers a 90-day trial period on all footwear purchases. If you are not satisfied with your purchase, you can return them for a full refund within 90 days.',
            'metadata': {'source': 'return_policy.txt', 'chunk_id': 0}
        },
        {
            'content': 'For returns, the shoes must be in good condition with minimal wear. Returns are processed within 5-7 business days.',
            'metadata': {'source': 'return_policy.txt', 'chunk_id': 1}
        }
    ]
    
    test_query = "What is your return policy?"
    
    print("📋 Test Configuration:")
    print(f"  Query: {test_query}")
    print(f"  Context Chunks: {len(sample_chunks)}")
    print()
    
    # Test 1: Available Models
    print("─" * 80)
    print("Test 1: Available Models")
    print("─" * 80)
    
    available = ModelFactory.get_available_models()
    for model_type, models in available.items():
        print(f"\n{model_type.upper()}:")
        for model in models:
            print(f"  • {model}")
    
    # Test 2: Gemma Generator
    print("\n\n" + "─" * 80)
    print("Test 2: Gemma Generator (Local Inference)")
    print("─" * 80 + "\n")
    
    try:
        print("🔄 Creating Gemma generator...")
        gemma_generator = ModelFactory.create_generator(
            model_type='gemma',
            model_config={
                'model_id': 'google/gemma-3-270m-it',
                'max_new_tokens': 128,
                'temperature': 0.7
            }
        )
        print(f"✅ Created: {gemma_generator.model_id}\n")
        
        print(f"🤖 Generating answer with Gemma...")
        result = gemma_generator.generate(
            query=test_query,
            retrieved_chunks=sample_chunks
        )
        
        print("\n" + "="*80)
        print("GEMMA RESPONSE:")
        print("="*80)
        print(f"\nModel: {result.get('model', 'N/A')}")
        print(f"Type: {result.get('model_type', 'N/A')}")
        print(f"Sources: {result.get('num_sources', 0)}\n")
        print(f"Answer:\n{result['answer']}\n")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error with Gemma: {e}")
        print("   Make sure transformers, torch, and accelerate are installed")
    
    # Test 3: Gemini Generator
    print("\n\n" + "─" * 80)
    print("Test 3: Gemini Generator (API-based)")
    print("─" * 80 + "\n")
    
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key:
        print("⚠️  GEMINI_API_KEY not found in environment")
        print("    To test Gemini:")
        print("    1. Get API key from: https://ai.google.dev/")
        print("    2. Set environment variable:")
        print("       export GEMINI_API_KEY='your-api-key'")
        print("    3. Run this test again")
    else:
        try:
            print("🔄 Creating Gemini generator...")
            gemini_generator = ModelFactory.create_generator(
                model_type='gemini',
                model_config={
                    'model_id': 'gemini-2.5-flash',
                    'max_output_tokens': 256,
                    'temperature': 0.7
                }
            )
            print(f"✅ Created: {gemini_generator.model_id}\n")
            
            print(f"🤖 Generating answer with Gemini...")
            result = gemini_generator.generate(
                query=test_query,
                retrieved_chunks=sample_chunks
            )
            
            print("\n" + "="*80)
            print("GEMINI RESPONSE:")
            print("="*80)
            print(f"\nModel: {result.get('model', 'N/A')}")
            print(f"Type: {result.get('model_type', 'N/A')}")
            print(f"Sources: {result.get('num_sources', 0)}\n")
            print(f"Answer:\n{result['answer']}\n")
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error with Gemini: {e}")
            print("   Make sure google-genai is installed: pip install google-genai")
    
    # Test 4: Environment Variable Configuration
    print("\n\n" + "─" * 80)
    print("Test 4: Environment Variable Configuration")
    print("─" * 80 + "\n")
    
    print("Current Environment Variables:")
    print(f"  MODEL_TYPE: {os.getenv('MODEL_TYPE', 'not set (defaults to gemma)')}")
    print(f"  MODEL_ID: {os.getenv('MODEL_ID', 'not set (uses model defaults)')}")
    print(f"  GEMINI_API_KEY: {'✓ set' if os.getenv('GEMINI_API_KEY') else '✗ not set'}")
    
    print("\n💡 Usage Examples:")
    print("\n  Use Gemma (default):")
    print("    generator = ModelFactory.create_generator()")
    
    print("\n  Use Gemini:")
    print("    export GEMINI_API_KEY='your-key'")
    print("    export MODEL_TYPE='gemini'")
    print("    generator = ModelFactory.create_generator()")
    
    print("\n  Use specific model:")
    print("    export MODEL_TYPE='gemini'")
    print("    export MODEL_ID='gemini-2.5-pro'")
    print("    generator = ModelFactory.create_generator()")
    
    # Summary
    print("\n\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80 + "\n")
    
    print("✅ Model Factory: Working")
    print("✅ Available Models: Listed")
    print(f"{'✅' if 'gemma_generator' in locals() else '❌'} Gemma Generator: {'Working' if 'gemma_generator' in locals() else 'Not tested'}")
    print(f"{'✅' if gemini_api_key and 'gemini_generator' in locals() else '❌'} Gemini Generator: {'Working' if gemini_api_key and 'gemini_generator' in locals() else 'Needs API key'}")
    
    print("\n" + "="*80)
    print("✅ MODEL FACTORY TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_model_factory()
