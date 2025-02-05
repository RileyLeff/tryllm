from semanticscholar import SemanticScholar
import os
from dotenv import load_dotenv
import json
from pprint import pprint

def safe_serialize(obj):
    """Helper function to serialize objects with potential circular references"""
    try:
        return {
            key: value for key, value in obj.__dict__.items()
            if not key.startswith('_')
        }
    except:
        return str(obj)

def inspect_citation_structure(doi: str):
    """Inspect the structure of citation data for a given DOI"""
    load_dotenv()
    api_key = os.getenv('SEMANTIC_API_KEY')
    sch = SemanticScholar(api_key=api_key)
    
    print(f"\nFetching paper with DOI: {doi}")
    paper = sch.get_paper(doi)
    print("\nPaper structure:")
    pprint(vars(paper))
    
    print("\nFetching citations...")
    citations = sch.get_paper_citations(paper.paperId)
    
    if citations:
        print(f"\nFound {len(citations)} citations")
        print("\nFirst citation structure:")
        pprint(vars(citations[0]))
        
        # Try to access different potential paths to DOI
        first_citation = citations[0]
        print("\nPotential DOI locations:")
        if hasattr(first_citation, '_data'):
            print("._data exists")
            if 'citingPaper' in first_citation._data:
                print("._data['citingPaper'] exists")
                citing_paper = first_citation._data['citingPaper']
                if 'externalIds' in citing_paper:
                    print("._data['citingPaper']['externalIds']:", citing_paper['externalIds'])

# Test with a few of your DOIs
test_dois = [
    "10.1111/nph.18275",  # one of your papers that was failing
    # Add 1-2 more DOIs if you want to compare
]

for doi in test_dois:
    inspect_citation_structure(doi)