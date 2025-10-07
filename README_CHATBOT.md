# Interactive Chatbot - Usage Guide

## Overview
The interactive chatbot (`chatbot_interactive.py`) allows you to test the LangGraph orchestrator by typing queries in real-time, just like chatting with a customer service bot.

## How to Run

```bash
# Activate virtual environment
source venv/bin/activate

# Run the interactive chatbot
python examples/chatbot_interactive.py
```

## Features

### 🎯 Automatic Query Routing
The chatbot automatically routes your queries to the appropriate agent:
- **🔍 RETRIEVAL** - Documentation, policies, FAQs, procedures
- **💬 CONVERSATIONAL** - Greetings, thanks, casual conversation
- **📞 API CALL** - Real-time data (weather, order tracking, live status)
- **⚠️ COMPLAINT** - Problems, issues, broken products, complaints

### 📊 Session Statistics
Track how many queries you've made and which categories they fell into.

### 💡 Special Commands

| Command | Description |
|---------|-------------|
| `help` | Show help information |
| `stats` | Display session statistics |
| `examples` | Show example queries for each category |
| `quit`, `exit`, `bye` | End the session |

## Example Session

```
👤 You: Hello, how are you?

🎯 ROUTING RESULT:
   💬 Category: CONVERSATIONAL
   📊 Confidence: 🟢 HIGH
   💭 Reasoning: BERT semantic similarity: 0.916

💬 CONVERSATIONAL AGENT ACTIVATED
   → Processing your message...
   → Happy to chat with you!

👤 You: What is your return policy?

🎯 ROUTING RESULT:
   🔍 Category: RETRIEVAL
   📊 Confidence: 🟢 HIGH
   💭 Reasoning: BERT semantic similarity: 0.954

🔍 RETRIEVAL AGENT ACTIVATED
   → Searching documentation and knowledge base...
   → I can help you find information about our policies and procedures.

👤 You: My product arrived broken!

🎯 ROUTING RESULT:
   ⚠️ Category: COMPLAINT
   📊 Confidence: 🟢 HIGH
   💭 Reasoning: Detected complaint-related keywords

⚠️ COMPLAINT AGENT ACTIVATED
   → Escalating to customer service team...
   → We apologize for the inconvenience. A specialist will assist you.

👤 You: quit

👋 Thanks for chatting! Here are your session statistics:
   📊 Total queries: 3
   🔍 Retrieval: 1
   💬 Conversational: 1
   📞 API Call: 0
   ⚠️ Complaint: 1
```

## Technical Details

### Model Used
- **BERT Base Uncased** for semantic similarity routing
- Pre-computed category embeddings for fast inference
- Hybrid approach: BERT embeddings + keyword overrides

### Routing Confidence
- **🟢 HIGH**: Strong match with category
- **🟡 MEDIUM**: Moderate match
- **🔴 LOW**: Uncertain match (uses keyword fallback)

### Performance
- First load: ~15 seconds (BERT model loading)
- Subsequent queries: < 1 second per query
- Accuracy: 100% on test suite

## Tips for Best Results

1. **Be natural**: Type as you would in a real chat
2. **Be specific**: Clear questions get better routing
3. **Try edge cases**: Test with ambiguous queries
4. **Check stats**: Use the `stats` command to see routing patterns

## Example Queries to Try

### Retrieval Queries
- "What is your shipping policy?"
- "How do I reset my password?"
- "Where can I find warranty information?"
- "Tell me about your privacy policy"

### Conversational Queries
- "Hello!"
- "Thanks for your help!"
- "Good morning"
- "Have a great day!"

### API Call Queries
- "What's the weather in New York?"
- "Track my order #12345"
- "Check delivery status"
- "Get current stock price"

### Complaint Queries
- "My product arrived broken!"
- "This is unacceptable!"
- "I'm very frustrated with this service"
- "The item is defective"

## Troubleshooting

### Model Loading is Slow
- First time: BERT downloads (~440MB)
- Subsequent runs: Loads from cache (~15 seconds)
- Normal behavior on CPU

### KeyboardInterrupt (Ctrl+C)
- Use to exit at any time
- Statistics will be displayed before exit

### Import Errors
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt`

## Development

The chatbot uses:
- `LangGraphOrchestrator` for query routing
- BERT embeddings for semantic similarity
- Keyword-based fallback for edge cases
- Real-time user interaction via `input()`

Modify the routing logic in `src/router/router_node.py` to customize behavior.
