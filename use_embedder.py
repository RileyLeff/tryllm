from project import ZoteroEmbedder
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the embedder with your Zotero storage path
storage_path = os.path.expanduser("~/Zotero/storage")
embedder = ZoteroEmbedder(
    storage_path=storage_path,
    persist_directory='chroma_db'
)

# Test PDF access first (optional)
print("Testing PDF access...")
embedder.test_pdf_access(limit=3)

# Create embeddings
print("\nCreating embeddings...")
stats = embedder.create_embeddings(
    chunk_size=1000,  # Characters per chunk
    overlap=100,      # Characters of overlap between chunks
    batch_size=64     # How many chunks to process at once
)

# Print stats
print("\nFinal Statistics:")
print(f"Processed PDFs: {stats['processed_pdfs']}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Average chunks per document: {stats['avg_chunks_per_doc']:.2f}")
print(f"Skipped items: {stats['skipped_items']}")

# Example search
query = "What are the main drivers of drought-induced tree mortality?"
print(f"\nTesting search with query: {query}")
results = embedder.search(query, n_results=3)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Title: {result['metadata']['title']}")
    print(f"Authors: {result['metadata']['authors']}")  # Now just print the string
    print(f"Year: {result['metadata']['year']}")
    print(f"Relevance score: {1 - result['distance']:.3f}")
    print(f"Excerpt: {result['chunk'][:200]}...")

# Example search
query2 = "What are key challenges developing process models of plant physiology?"
print(f"\nTesting search with query: {query2}")
results = embedder.search(query2, n_results=3)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Title: {result['metadata']['title']}")
    print(f"Authors: {result['metadata']['authors']}")  # Now just print the string
    print(f"Year: {result['metadata']['year']}")
    print(f"Relevance score: {1 - result['distance']:.3f}")
    print(f"Excerpt: {result['chunk'][:500]}...")