"""Router prompt templates for query classification."""

ROUTING_PROMPT_TEMPLATE = """Classify this query into ONE category:

Categories:
1. complaint - Problems, complaints, negative feedback, dissatisfaction
2. api_call - Real-time data (weather, order status, stock prices)
3. retrieval - Documentation, policies, procedures, static company info
4. conversational - Greetings, thanks, casual chat, small talk

Priority: complaint > api_call > retrieval > conversational

Query: {user_input}

Analysis:
- Is there a problem or complaint? → complaint
- Need real-time/current data? → api_call
- Need company docs/policies? → retrieval
- Just casual conversation? → conversational

Respond in this format:
Category: [category]
Reasoning: [brief explanation]
Confidence: [high/medium/low]
"""


FALLBACK_ROUTING_PROMPT = """You are a query classifier. Classify this query strictly:

Query: {user_input}

Choose ONE:
- complaint: Customer is unhappy, has a problem, or reporting an issue
- api_call: Needs live data (weather, tracking, current status)
- retrieval: Needs company info (policies, docs, procedures, FAQs)
- conversational: Greeting, thanks, or casual chat

Category:"""
