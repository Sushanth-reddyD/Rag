"""Test the fine-tuned BERT router model."""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os


class FineTunedRouter:
    """Router using fine-tuned BERT model."""
    
    def __init__(self, model_path: str = "./models/fine_tuned_router"):
        """Initialize with fine-tuned model."""
        print(f"🔄 Loading fine-tuned model from {model_path}...")
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first using train_router.py")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Load label mappings
        with open(f"{model_path}/label_mappings.json", 'r') as f:
            mappings = json.load(f)
        
        # Convert string keys back to integers for id_to_category
        self.id_to_category = {int(k): v for k, v in mappings['id_to_category'].items()}
        self.category_to_id = mappings['category_to_id']
        
        # Set device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ Using Mac GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("✅ Using NVIDIA GPU (CUDA)")
        else:
            self.device = "cpu"
            print("⚠️ Using CPU")
        
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
    
    def predict(self, text: str, return_probs: bool = False):
        """Predict category for a given text."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get predicted class
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()
        category = self.id_to_category[pred_id]
        
        result = {
            'category': category,
            'confidence': confidence,
            'confidence_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low'
        }
        
        if return_probs:
            # Get all probabilities
            all_probs = {self.id_to_category[i]: probs[0][i].item() for i in range(len(self.id_to_category))}
            result['probabilities'] = all_probs
        
        return result


def run_test_queries(router):
    """Run test queries through the fine-tuned router."""
    
    test_queries = [
        # Retrieval queries
        ("which product do you recommend?", "retrieval"),
        ("How do I reset my password?", "retrieval"),
        ("what places do you ship outside australia?", "retrieval"),
        ("Can i return the product i bought it not from india in india?", "retrieval"),
        ("Tell me about your company's privacy policy", "retrieval"),
        ("How do I submit a refund request?", "retrieval"),
        ("What is your return policy?", "retrieval"),
        ("I need the documentation for returns", "retrieval"),
        
        # Conversational queries
        ("Hello, how are you?", "conversational"),
        ("Thanks for your help!", "conversational"),
        ("Good morning!", "conversational"),
        ("Have a great day!", "conversational"),
        
        # API call queries
        ("What's the weather in London?", "api_call"),
        ("Track my order #12345", "api_call"),
        ("Where is my delivery right now?", "api_call"),
        ("What's the current stock price?", "api_call"),
        
        # Complaint queries
        ("My product arrived broken!", "complaint"),
        ("I've been waiting for 2 weeks, this is unacceptable!", "complaint"),
        ("The quality is terrible!", "complaint"),
        ("Can you help me understand your return policy? Mine is defective.", "complaint"),
    ]
    
    print("\n" + "="*100)
    print("🧪 TESTING FINE-TUNED MODEL")
    print("="*100)
    
    correct = 0
    total = len(test_queries)
    
    for i, (query, expected) in enumerate(test_queries, 1):
        result = router.predict(query, return_probs=True)
        predicted = result['category']
        confidence = result['confidence']
        
        is_correct = predicted == expected
        if is_correct:
            correct += 1
            status = "✅ CORRECT"
        else:
            status = f"❌ INCORRECT (expected: {expected})"
        
        print(f"\n{'─'*100}")
        print(f"Query {i}/{total}: {query}")
        print(f"{'─'*100}")
        print(f"Predicted: {predicted} | Confidence: {confidence:.4f} ({result['confidence_level']})")
        print(f"Expected:  {expected}")
        print(f"Status:    {status}")
        
        # Show probabilities for all categories
        print("\nProbabilities:")
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for cat, prob in sorted_probs:
            bar_length = int(prob * 50)
            bar = "█" * bar_length
            print(f"  {cat:15s} {prob:.4f} {bar}")
    
    accuracy = (correct / total) * 100
    
    print("\n" + "="*100)
    print(f"📊 RESULTS: {correct}/{total} correct")
    print(f"🎯 Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 85:
        print("✅ SUCCESS: Routing accuracy meets requirement (≥85%)")
    else:
        print("⚠️ WARNING: Routing accuracy below requirement (≥85%)")
    
    print("="*100)
    
    return accuracy


def interactive_mode(router):
    """Interactive testing mode."""
    print("\n" + "="*100)
    print("🎮 INTERACTIVE MODE")
    print("="*100)
    print("Type your queries to test the model. Commands: 'quit' to exit, 'help' for help")
    print("="*100)
    
    while True:
        try:
            query = input("\n👤 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Type any query to classify it")
                print("  - 'quit' or 'exit' to quit")
                print("  - 'help' to show this message")
                continue
            
            result = router.predict(query, return_probs=True)
            
            print(f"\n🤖 Prediction:")
            print(f"   Category:   {result['category']}")
            print(f"   Confidence: {result['confidence']:.4f} ({result['confidence_level']})")
            print(f"\n   Probabilities:")
            sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
            for cat, prob in sorted_probs:
                bar_length = int(prob * 30)
                bar = "█" * bar_length
                print(f"     {cat:15s} {prob:.4f} {bar}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test fine-tuned router model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/fine_tuned_router",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Load model
    router = FineTunedRouter(model_path=args.model_path)
    
    # Run tests
    if not args.interactive:
        run_test_queries(router)
        
        # Ask if user wants interactive mode
        response = input("\n🎮 Would you like to try interactive mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_mode(router)
    else:
        interactive_mode(router)


if __name__ == "__main__":
    main()
