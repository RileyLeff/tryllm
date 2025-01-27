from transformers import AutoTokenizer
import numpy as np

def analyze_embedding_tokens(embedder):
    # Get the tokenizer matching your sentence-transformer model
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    # Get all documents from ChromaDB
    all_docs = embedder.collection.get()
    
    # Count tokens for each chunk
    token_counts = [len(tokenizer.encode(doc)) for doc in all_docs['documents']]
    
    # Calculate statistics
    stats = {
        'total_chunks': len(token_counts),
        'total_tokens': sum(token_counts),
        'avg_tokens_per_chunk': np.mean(token_counts),
        'min_tokens': min(token_counts),
        'max_tokens': max(token_counts),
        'median_tokens': np.median(token_counts)
    }
    
    print(f"Total chunks: {stats['total_chunks']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
    print(f"Token range: {stats['min_tokens']} - {stats['max_tokens']}")
    print(f"Median tokens: {stats['median_tokens']:.1f}")
    
    # If using OpenAI's pricing
    print(f"\nEstimated OpenAI embedding cost: ${(stats['total_tokens'] * 0.00002):,.2f}")
    
    return stats

# Use it with your existing embedder
embedder = ZoteroEmbedder(storage_path="~/Zotero/storage")
stats = analyze_embedding_tokens(embedder)