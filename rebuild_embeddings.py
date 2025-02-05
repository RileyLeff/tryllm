import os
from pathlib import Path
import toml
import argparse
from project import ZoteroEmbedder
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ChunkingParams:
    words_per_token: float
    max_token_utilization: float
    overlap_ratio: float

def load_config():
    config_path = Path("embedding_config.toml")
    if not config_path.exists():
        raise FileNotFoundError("embedding_config.toml not found")
    return toml.load(config_path)

def calculate_chunk_size(
    model_config: dict,
    chunking_config: dict,
    params: ChunkingParams
) -> Tuple[int, int]:
    """Calculate appropriate chunk size based on model's max tokens and chunking parameters"""
    max_tokens = model_config.get('max_tokens', chunking_config['default_chunk_size'])
    
    chunk_size = int(max_tokens * params.words_per_token * params.max_token_utilization)
    final_chunk_size = max(chunk_size, chunking_config['default_chunk_size'])
    overlap = int(final_chunk_size * params.overlap_ratio)
    
    return final_chunk_size, overlap

def main():
    parser = argparse.ArgumentParser(description='Rebuild embeddings for Zotero papers')
    parser.add_argument('--mode', 
                       choices=['new_only', 'update_all', 'from_scratch'],
                       required=True,
                       help='Embedding update mode: new_only (add new papers only), update_all (reprocess existing and add new), from_scratch (delete and rebuild)')
    parser.add_argument('--model', 
                       choices=['minilm', 'mpnet', 'bge-large'],
                       required=True,
                       help='Model to use for embeddings')
    parser.add_argument('--storage-path', 
                       default='~/Zotero/storage',
                       help='Path to Zotero storage')
    parser.add_argument('--force-chunk-size', 
                       type=int,
                       help='Override automatic chunk size calculation')
    parser.add_argument('--force-overlap', 
                       type=int,
                       help='Override automatic overlap calculation')
    parser.add_argument('--words-per-token', 
                       type=float, 
                       default=0.75,
                       help='Average number of words per token')
    parser.add_argument('--max-token-utilization', 
                       type=float, 
                       default=0.9,
                       help='Maximum fraction of model\'s token limit to use')
    parser.add_argument('--overlap-ratio', 
                       type=float, 
                       default=0.2,
                       help='Fraction of chunk size to use as overlap')
    
    args = parser.parse_args()
    
    # Confirm destructive operations
    if args.mode in ['from_scratch', 'update_all']:
        print(f"\nWARNING: You are about to {args.mode} the embedding database.")
        response = input("Are you sure you want to continue? [y/N]: ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    chunking_params = ChunkingParams(
        words_per_token=args.words_per_token,
        max_token_utilization=args.max_token_utilization,
        overlap_ratio=args.overlap_ratio
    )
    
    config = load_config()
    model_config = config['models'][args.model]
    storage_config = config['storage']
    chunking_config = config['chunking']
    
    if args.force_chunk_size:
        chunk_size = args.force_chunk_size
        overlap = args.force_overlap or chunking_config['default_overlap']
    else:
        chunk_size, overlap = calculate_chunk_size(
            model_config, 
            chunking_config, 
            chunking_params
        )
    
    print(f"\nConfiguration:")
    print(f"Model: {model_config['name']}")
    print(f"Chunk size: {chunk_size} words")
    print(f"Overlap: {overlap} words")
    print(f"Words per token: {chunking_params.words_per_token}")
    print(f"Max token utilization: {chunking_params.max_token_utilization}")
    print(f"Overlap ratio: {chunking_params.overlap_ratio}")
    
    # Handle database preparation based on mode
    if args.mode == 'from_scratch':
        print("\nClearing existing embeddings...")
        import shutil
        if os.path.exists(storage_config['default_persist_directory']):
            shutil.rmtree(storage_config['default_persist_directory'])
    
    # Initialize embedder after potential database clearing
    embedder = ZoteroEmbedder(
        storage_path=os.path.expanduser(args.storage_path),
        persist_directory=storage_config['default_persist_directory']
    )
    
    print(f"\nCreating embeddings in {args.mode} mode...")
    
    stats = embedder.create_embeddings(
        chunk_size=chunk_size,
        overlap=overlap,
        update_mode=args.mode
    )
    
    print("\nEmbedding creation complete!")
    print(f"Processed PDFs: {stats['processed_pdfs']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Average chunks per document: {stats['avg_chunks_per_doc']:.2f}")
    print(f"Skipped items: {stats['skipped_items']}")
    
    print("\nVerifying database contents...")
    verify_stats = embedder.get_collection_stats()
    print(f"Total documents in database: {verify_stats['total_documents']}")
    print(f"Total chunks in database: {verify_stats['total_chunks']}")

if __name__ == "__main__":
    main()