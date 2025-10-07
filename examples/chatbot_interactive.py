"""Interactive chatbot to test the LangGraph orchestrator with user input."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.router.orchestrator import LangGraphOrchestrator


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*80)
    print("🤖 INTERACTIVE CHATBOT - LangGraph Orchestrator")
    print("="*80)
    print("\n📝 This chatbot will route your queries to the appropriate agent:")
    print("   • 🔍 RETRIEVAL - Documentation, policies, FAQs")
    print("   • 💬 CONVERSATIONAL - Greetings, thanks, casual chat")
    print("   • 📞 API CALL - Real-time data (weather, order tracking)")
    print("   • ⚠️  COMPLAINT - Problems, issues, negative feedback")
    print("\n💡 Type your message and press Enter. Type 'quit', 'exit', or 'bye' to stop.")
    print("="*80 + "\n")


def print_routing_result(result):
    """Print the routing result in a formatted way."""
    category = result["routing_decision"]
    reasoning = result["reasoning"]
    confidence = result["confidence"]
    
    # Category emoji mapping
    category_emoji = {
        "retrieval": "🔍",
        "conversational": "💬",
        "api_call": "📞",
        "complaint": "⚠️"
    }
    
    # Confidence color (using text)
    confidence_display = {
        "high": "🟢 HIGH",
        "medium": "🟡 MEDIUM",
        "low": "🔴 LOW"
    }
    
    emoji = category_emoji.get(category, "❓")
    conf_display = confidence_display.get(confidence, confidence.upper())
    
    print("\n" + "-"*80)
    print(f"🎯 ROUTING RESULT:")
    print(f"   {emoji} Category: {category.upper()}")
    print(f"   📊 Confidence: {conf_display}")
    print(f"   💭 Reasoning: {reasoning}")
    print("-"*80)
    
    # Simulate agent response based on category
    print(f"\n{emoji} {category.upper()} AGENT ACTIVATED")
    
    if category == "retrieval":
        print("   → Searching documentation and knowledge base...")
        print("   → I can help you find information about our policies and procedures.")
    elif category == "conversational":
        print("   → Processing your message...")
        print("   → Happy to chat with you!")
    elif category == "api_call":
        print("   → Calling external API for real-time data...")
        print("   → Fetching the latest information for you.")
    elif category == "complaint":
        print("   → Escalating to customer service team...")
        print("   → We apologize for the inconvenience. A specialist will assist you.")
    
    print()


def main():
    """Run interactive chatbot."""
    print_banner()
    
    # Initialize orchestrator
    print("🔄 Initializing orchestrator (this may take a moment)...")
    try:
        orchestrator = LangGraphOrchestrator()
        print("✅ Orchestrator ready!\n")
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return
    
    # Track conversation statistics
    total_queries = 0
    category_counts = {"retrieval": 0, "conversational": 0, "api_call": 0, "complaint": 0}
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "bye", "goodbye", "q"]:
                print("\n" + "="*80)
                print("👋 Thanks for chatting! Here are your session statistics:")
                print(f"   📊 Total queries: {total_queries}")
                print(f"   🔍 Retrieval: {category_counts['retrieval']}")
                print(f"   💬 Conversational: {category_counts['conversational']}")
                print(f"   📞 API Call: {category_counts['api_call']}")
                print(f"   ⚠️  Complaint: {category_counts['complaint']}")
                print("="*80)
                print("Goodbye! 👋\n")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Special commands
            if user_input.lower() == "help":
                print("\n📖 HELP:")
                print("   • Just type your question or message naturally")
                print("   • The system will automatically route it to the right agent")
                print("   • Type 'stats' to see session statistics")
                print("   • Type 'examples' to see example queries")
                print("   • Type 'quit' or 'exit' to end the session\n")
                continue
            
            if user_input.lower() == "stats":
                print("\n📊 SESSION STATISTICS:")
                print(f"   Total queries: {total_queries}")
                print(f"   🔍 Retrieval: {category_counts['retrieval']}")
                print(f"   💬 Conversational: {category_counts['conversational']}")
                print(f"   📞 API Call: {category_counts['api_call']}")
                print(f"   ⚠️  Complaint: {category_counts['complaint']}\n")
                continue
            
            if user_input.lower() == "examples":
                print("\n💡 EXAMPLE QUERIES:")
                print("   🔍 Retrieval:")
                print("      - What is your return policy?")
                print("      - How do I reset my password?")
                print("   💬 Conversational:")
                print("      - Hello, how are you?")
                print("      - Thanks for your help!")
                print("   📞 API Call:")
                print("      - What's the weather in London?")
                print("      - Track my order #12345")
                print("   ⚠️  Complaint:")
                print("      - My product arrived broken!")
                print("      - This is unacceptable!\n")
                continue
            
            # Route the query
            total_queries += 1
            result = orchestrator.route_query(user_input)
            
            # Update statistics
            category = result["routing_decision"]
            if category in category_counts:
                category_counts[category] += 1
            
            # Display result
            print_routing_result(result)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user (Ctrl+C)")
            print("👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
