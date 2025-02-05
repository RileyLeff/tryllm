from project import ZoteroEmbedder
from research_assistant import ResearchAssistant, QueryConfig, AnalysisConfig, ComparisonConfig
import os
from dotenv import load_dotenv
from llm_providers import MODEL_CONFIGS
import argparse
from typing import List

def list_available_models() -> List[str]:
    print("\nAvailable models:")
    for name, config in MODEL_CONFIGS.items():
        print(f"- {name}: {config['description']}")
    return list(MODEL_CONFIGS.keys())

def main():
    parser = argparse.ArgumentParser(description='Research Assistant CLI')
    parser.add_argument('--mode',
                       choices=['query', 'analyze', 'compare'],
                       required=True,
                       help='Operation mode')
    parser.add_argument('--model',
                       default='claude-sonnet',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='LLM model to use')
    parser.add_argument('--storage-path',
                       default='~/Zotero/storage',
                       help='Path to Zotero storage')
    parser.add_argument('--question',
                       help='Research question to ask (for query mode)')
    parser.add_argument('--title',
                       help='Paper title to analyze (for analyze mode)')
    parser.add_argument('--titles',
                       nargs='+',
                       help='Paper titles to compare (for compare mode)')
    parser.add_argument('--n-chunks',
                       type=int,
                       default=5,
                       help='Number of relevant chunks to include')
    parser.add_argument('--max-chars',
                       type=int,
                       default=8000,
                       help='Maximum characters per chunk/paper')
    parser.add_argument('--system-prompt',
                       help='Override default system prompt')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize embedder
    embedder = ZoteroEmbedder(
        storage_path=os.path.expanduser(args.storage_path),
        persist_directory='chroma_db'
    )
    
    # Initialize assistant
    assistant = ResearchAssistant(embedder, model_name=args.model)
    
    # Create appropriate config based on mode
    if args.mode == 'query':
        if not args.question:
            parser.error("--question is required for query mode")
        config = QueryConfig(
            n_chunks=args.n_chunks,
            max_chars_per_chunk=args.max_chars,
            system_prompt=args.system_prompt or QueryConfig().system_prompt
        )
        result = assistant.query(args.question, config)
        
    elif args.mode == 'analyze':
        if not args.title:
            parser.error("--title is required for analyze mode")
        config = AnalysisConfig(
            max_chars=args.max_chars,
            system_prompt=args.system_prompt or AnalysisConfig().system_prompt
        )
        result = assistant.analyze_paper(args.title, config)
        
    elif args.mode == 'compare':
        if not args.titles or len(args.titles) < 2:
            parser.error("--titles with at least 2 papers is required for compare mode")
        config = ComparisonConfig(
            max_chars_per_paper=args.max_chars,
            system_prompt=args.system_prompt or ComparisonConfig().system_prompt
        )
        result = assistant.compare_papers(args.titles, config)
    
    print(result)

if __name__ == "__main__":
    main()