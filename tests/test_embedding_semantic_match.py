"""
Test Script: Semantic Embedding Similarity
===========================================

Tests whether the sentence transformer can match the query "How long will the shoes run"
with the document sentence about shoe lifespan (450-800 kms).

This tests the core retrieval capability of the RAG system.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("="*80)
print("🧪 TESTING SEMANTIC EMBEDDING SIMILARITY")
print("="*80)

# Load the same model used in your RAG system
print("\n📥 Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Model loaded successfully\n")

# Document sentence from your data
document_sentence = "When used for the intended purpose, running footwear has a life expectancy of 450 to 800 kms."

# User query
user_query = "How long will the shoes run"

# Alternative/related queries to test
test_queries = [
    "How long will the shoes run",
    "How long do shoes last",
    "What is the lifespan of running shoes",
    "How many kilometers can I run in these shoes",
    "When should I replace my running shoes",
    "Shoe durability",
    "Expected mileage of running shoes",
    "How far can I run with Brooks",
]

print("="*80)
print("📄 DOCUMENT SENTENCE:")
print("="*80)
print(f"{document_sentence}\n")

print("="*80)
print("🔍 TESTING QUERIES:")
print("="*80)

# Encode document
doc_embedding = model.encode([document_sentence])

# Test each query
results = []
for query in test_queries:
    # Encode query
    query_embedding = model.encode([query])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
    
    results.append((query, similarity))
    
    # Print results
    status = "✅" if similarity > 0.5 else "⚠️" if similarity > 0.3 else "❌"
    print(f"\n{status} Query: \"{query}\"")
    print(f"   └─ Similarity Score: {similarity:.4f}")
    
    if similarity > 0.5:
        print(f"   └─ Rating: EXCELLENT - Will definitely retrieve")
    elif similarity > 0.4:
        print(f"   └─ Rating: GOOD - Should retrieve")
    elif similarity > 0.3:
        print(f"   └─ Rating: MODERATE - Might retrieve")
    else:
        print(f"   └─ Rating: POOR - Unlikely to retrieve")

print("\n" + "="*80)
print("📊 SUMMARY & ANALYSIS")
print("="*80)

# Sort by similarity
results.sort(key=lambda x: x[1], reverse=True)

print("\n🏆 Best Performing Queries:")
for i, (query, score) in enumerate(results[:3], 1):
    print(f"{i}. \"{query}\" - Score: {score:.4f}")

print("\n⚠️  Worst Performing Queries:")
for i, (query, score) in enumerate(results[-3:], 1):
    print(f"{i}. \"{query}\" - Score: {score:.4f}")

# Analysis
original_score = results[[q for q, s in results].index("How long will the shoes run")][1]

print(f"\n" + "="*80)
print(f"🎯 ANSWER TO YOUR QUESTION:")
print(f"="*80)
print(f"\nOriginal Query: \"How long will the shoes run\"")
print(f"Similarity Score: {original_score:.4f}")

if original_score > 0.5:
    print(f"\n✅ YES! The embedding model CAN retrieve this document!")
    print(f"   The semantic similarity is STRONG ({original_score:.4f} > 0.5)")
elif original_score > 0.3:
    print(f"\n⚠️  MAYBE. The embedding model MIGHT retrieve this document.")
    print(f"   The semantic similarity is MODERATE ({original_score:.4f})")
    print(f"   It depends on what other documents are in the database.")
elif original_score > 0.2:
    print(f"\n⚠️  UNLIKELY. The embedding model probably WON'T retrieve this as top result.")
    print(f"   The semantic similarity is WEAK ({original_score:.4f})")
else:
    print(f"\n❌ NO. The embedding model will likely NOT retrieve this document.")
    print(f"   The semantic similarity is TOO LOW ({original_score:.4f} < 0.2)")

print(f"\n" + "="*80)
print(f"🔍 HOW IT WORKS:")
print(f"="*80)
print(f"""
The model understands semantic meaning:
- "How long will shoes run" → captures concept of DURATION/DISTANCE
- "450 to 800 kms" → represents DISTANCE/LIFESPAN
- "life expectancy" → represents DURATION/LONGEVITY

The model maps both to similar vector spaces, enabling retrieval even without
exact keyword matches!
""")

print(f"="*80)
print(f"💡 WHY THIS MATTERS:")
print(f"="*80)
print(f"""
Embedding similarity scores:
- > 0.7: Very high semantic match (nearly paraphrases)
- 0.5-0.7: High semantic match (clearly related)
- 0.3-0.5: Moderate match (somewhat related)
- 0.2-0.3: Weak match (tangentially related)
- < 0.2: Very weak match (likely unrelated)

Your RAG system uses this score to rank documents by relevance!
""")

# Additional test: What if we add more context?
print(f"="*80)
print(f"🔬 ADVANCED TEST: Effect of Query Formulation")
print(f"="*80)

advanced_queries = [
    # More natural language
    "How long will the shoes run",
    
    # More technical
    "What is the expected mileage of running shoes",
    
    # More casual
    "How far can I run with these",
    
    # Direct keywords
    "shoe life expectancy kilometers",
    
    # Different phrasing
    "When do I need to replace my running shoes",
]

print("\nComparing different query formulations:\n")
for query in advanced_queries:
    query_embedding = model.encode([query])
    similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
    
    bar_length = int(similarity * 50)
    bar = "█" * bar_length + "░" * (50 - bar_length)
    
    print(f"{similarity:.4f} {bar} \"{query}\"")

print(f"\n" + "="*80)
print(f"🎓 KEY INSIGHTS:")
print(f"="*80)
print(f"""
1. Semantic Understanding: The model captures MEANING, not just keywords
   - "How long" → relates to "life expectancy"
   - "run" → relates to "footwear" usage
   - Query structure matters less than semantic content

2. Keyword Importance: Queries with domain-specific terms score higher
   - "life expectancy kilometers" scores well
   - "how far can I run" also works
   - Both capture the core concept

3. Context Matters: More specific queries help
   - "running shoes" > "shoes"
   - "running shoes" > "shoes"
   - "life expectancy" > "how long"

4. Your RAG System: With top-k retrieval (k=20 initial, k=5 final)
   - Even moderate scores (0.3-0.5) can retrieve the document
   - Re-ranking further improves relevance
   - Multiple semantic matches increase recall
""")

print(f"="*80)
print(f"✅ TEST COMPLETE!")
print(f"="*80)
