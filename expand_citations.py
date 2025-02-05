from semanticscholar import SemanticScholar
from pyzotero import zotero
import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Dict, List, Optional
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import argparse

class ZoteroAccessMode(Enum):
    LOCAL = "local"
    API = "api"
    BOTH = "both"

@dataclass
class ZoteroConfig:
    mode: ZoteroAccessMode
    zotero_dir: str  # Directory containing zotero.sqlite
    library_id: Optional[str] = None
    api_key: Optional[str] = None

class ZoteroAccessor:
    def __init__(self, config: ZoteroConfig):
        self.config = config
        self.api_client = None
        self.db_path = os.path.join(os.path.expanduser(config.zotero_dir), 'zotero.sqlite')
        
        if not os.path.exists(self.db_path):
            raise ValueError(f"Zotero database not found: {self.db_path}")
            
        if config.mode in [ZoteroAccessMode.API, ZoteroAccessMode.BOTH]:
            if not config.library_id or not config.api_key:
                raise ValueError("API mode requires library_id and api_key")
            self.api_client = zotero.Zotero(config.library_id, 'user', config.api_key)
    
    def get_api_dois(self) -> Dict[str, Dict]:
        """Get DOIs and metadata from Zotero API"""
        if self.api_client is None:
            return {}
            
        dois = {}
        start = 0
        limit = 100
        
        with tqdm(desc="Fetching from API") as pbar:
            while True:
                items = self.api_client.items(start=start, limit=limit)
                if not items:
                    break
                    
                for item in items:
                    doi = item['data'].get('DOI')
                    if doi:
                        dois[doi.lower()] = {
                            'title': item['data'].get('title', 'Unknown'),
                            'source': 'api'
                        }
                
                start += limit
                pbar.update(len(items))
        
        return dois
    
    def get_local_dois(self) -> Dict[str, Dict]:
        """Get DOIs from local Zotero database"""
        if self.config.mode not in [ZoteroAccessMode.LOCAL, ZoteroAccessMode.BOTH]:
            return {}
            
        dois = {}
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT fields.fieldName, itemDataValues.value, items.key, items.itemID
            FROM itemData
            JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
            JOIN fields ON itemData.fieldID = fields.fieldID
            JOIN items ON itemData.itemID = items.itemID
            WHERE fields.fieldName IN ('DOI', 'title')
            """
            
            items_data = defaultdict(dict)
            
            cursor = conn.execute(query)
            for field_name, value, key, item_id in cursor:
                items_data[item_id][field_name] = value
            
            for item_id, data in items_data.items():
                if 'DOI' in data:
                    dois[data['DOI'].lower()] = {
                        'title': data.get('title', 'Unknown'),
                        'source': 'local'
                    }
        
        return dois
    
    def get_all_dois(self) -> Dict[str, Dict]:
        """Get all DOIs based on configured access mode"""
        all_dois = {}
        
        if self.config.mode in [ZoteroAccessMode.API, ZoteroAccessMode.BOTH]:
            print("Getting DOIs from Zotero API...")
            api_dois = self.get_api_dois()
            print(f"Found {len(api_dois)} DOIs from API")
            all_dois.update(api_dois)
        
        if self.config.mode in [ZoteroAccessMode.LOCAL, ZoteroAccessMode.BOTH]:
            print("\nGetting DOIs from local database...")
            local_dois = self.get_local_dois()
            print(f"Found {len(local_dois)} DOIs from local database")
            all_dois.update(local_dois)
        
        print(f"Total unique DOIs: {len(all_dois)}")
        return all_dois
    
    def add_paper(self, paper: Dict):
        """Add a paper to Zotero"""
        if self.api_client is None:
            raise ValueError("Cannot add papers without API access")
            
        template = self.api_client.item_template('journalArticle')
        template['title'] = paper['title']
        template['DOI'] = paper['doi']
        template['date'] = str(paper['year']) if paper['year'] else ''
        
        # Add creators
        template['creators'] = []
        for author in paper['authors']:
            parts = author.split(' ')
            if len(parts) > 1:
                template['creators'].append({
                    'creatorType': 'author',
                    'firstName': ' '.join(parts[:-1]),
                    'lastName': parts[-1]
                })
            else:
                template['creators'].append({
                    'creatorType': 'author',
                    'lastName': author
                })
        
        # Add note about source
        template['notes'] = [{'note': f"Added via citation expansion from: {paper['source_paper']}"}]
        
        # Create the item
        self.api_client.create_items([template])

def collect_citation_dois(sch: SemanticScholar, source_doi: str, source_title: str, rate_limit: float) -> List[Dict]:
    """Get citation information for a single paper"""
    try:
        paper = sch.get_paper(source_doi)
        if not paper:
            return []
            
        citations = sch.get_paper_citations(paper.paperId)
        citation_data = []
        
        for citation in citations:
            if hasattr(citation, '_data') and 'citingPaper' in citation._data:
                citing_paper = citation._data['citingPaper']
                if 'externalIds' in citing_paper and 'DOI' in citing_paper['externalIds']:
                    citation_data.append({
                        'title': citing_paper.get('title', 'Unknown Title'),
                        'doi': citing_paper['externalIds']['DOI'].lower(),
                        'year': citing_paper.get('year'),
                        'authors': [
                            author.get('name', 'Unknown Author') 
                            for author in citing_paper.get('authors', [])
                        ],
                        'source_paper': source_title
                    })
        
        if citation_data:
            print(f"Found {len(citation_data)} citations with DOIs for {source_title}")
            
        if rate_limit > 0:
            time.sleep(rate_limit)
            
        return citation_data
    
    except Exception as e:
        print(f"\nError processing DOI {source_doi}: {str(e)}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Expand Zotero library with citations')
    parser.add_argument('--mode', 
                       choices=['local', 'api', 'both'],
                       default='local',
                       help='Zotero access mode')
    parser.add_argument('--zotero-dir',
                       default='~/Zotero',
                       help='Path to Zotero directory containing zotero.sqlite')
    parser.add_argument('--batch-size',
                       type=int,
                       default=50,
                       help='Number of papers to process before asking for confirmation')
    parser.add_argument('--rate-limit',
                       type=float,
                       default=1.1,
                       help='Time to wait between Semantic Scholar API calls (in seconds)')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize Semantic Scholar client with API key if available
    semantic_api_key = os.getenv('SEMANTIC_API_KEY')
    sch = SemanticScholar(api_key=semantic_api_key) if semantic_api_key else SemanticScholar()
    
    # Get Zotero credentials if needed
    mode = ZoteroAccessMode(args.mode)
    if mode in [ZoteroAccessMode.API, ZoteroAccessMode.BOTH]:
        library_id = os.getenv('LIBRARY_ID')
        api_key = os.getenv('API_KEY')
    else:
        library_id = None
        api_key = None
    
    # Initialize clients
    config = ZoteroConfig(
        mode=mode,
        zotero_dir=args.zotero_dir,
        library_id=library_id,
        api_key=api_key
    )
    
    zotero = ZoteroAccessor(config)
    
    # Get existing DOIs
    existing_papers = zotero.get_all_dois()
    existing_dois = set(existing_papers.keys())
    
    # Collect citations
    print("\nCollecting citations...")
    new_papers = []
    citation_sources = defaultdict(list)
    
        # In main(), update the citation collection loop:
    for doi, paper_info in tqdm(existing_papers.items(), desc="Processing papers"):
        print(f"\nProcessing: {paper_info['title']} (DOI: {doi})")
        try:
            citations = collect_citation_dois(sch, doi, paper_info['title'], args.rate_limit)
            if citations:
                print(f"Found {len(citations)} citations")
            for citation in citations:
                if citation['doi'] not in existing_dois:
                    new_papers.append(citation)
                    citation_sources[citation['doi']].append(paper_info['title'])
        except Exception as e:
            print(f"Failed to process {doi}: {str(e)}")
            continue
    
    # Remove duplicates while preserving source information
    unique_papers = {}
    for paper in new_papers:
        doi = paper['doi']
        if doi not in unique_papers:
            unique_papers[doi] = paper
    
    print(f"\nFound {len(unique_papers)} new papers to add")
    
    # Process in batches
    papers_list = list(unique_papers.values())
    for i in range(0, len(papers_list), args.batch_size):
        batch = papers_list[i:i + args.batch_size]
        
        print(f"\nBatch {i//args.batch_size + 1} of {len(papers_list)//args.batch_size + 1}")
        print("\nPapers to be added:")
        for j, paper in enumerate(batch, 1):
            print(f"\n{j}. {paper['title']}")
            print(f"   DOI: {paper['doi']}")
            print(f"   Cited by: {', '.join(citation_sources[paper['doi']])}")
        
        if mode != ZoteroAccessMode.LOCAL:
            response = input("\nAdd these papers to Zotero? [y/N]: ")
            if response.lower() == 'y':
                for paper in tqdm(batch, desc="Adding to Zotero"):
                    try:
                        zotero.add_paper(paper)
                        time.sleep(0.5)  # Be nice to the Zotero API
                    except Exception as e:
                        print(f"\nError adding paper {paper['doi']}: {str(e)}")
            else:
                print("Skipping batch.")
        else:
            print("\nRunning in local-only mode. Cannot add papers to Zotero without API access.")
    
    print("\nOperation complete!")

if __name__ == "__main__":
    main()