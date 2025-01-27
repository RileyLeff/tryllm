from pyzotero import zotero
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict
import fitz  # Change this line back
import os
from tqdm import tqdm
import chromadb
import uuid
import json
from dotenv import load_dotenv
import glob

# Load environment variables from .env file
load_dotenv()

class CustomEmbeddingFunction:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Filter out empty texts and keep track of their positions
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(input):
            if text and len(text.strip()) > 0:
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            # If no valid texts, return zero vectors
            embedding_dim = 384  # This is the dimension for 'all-MiniLM-L6-v2'
            return [[0.0] * embedding_dim] * len(input)

        # Get embeddings for valid texts
        with torch.no_grad():
            embeddings = self.model.encode(
                valid_texts,
                convert_to_tensor=True,
                device=self.device
            )
            embeddings = embeddings.cpu().numpy()

        # Create result array with zero vectors for invalid texts
        result = [[0.0] * embeddings.shape[1]] * len(input)
        
        # Put valid embeddings back in their original positions
        for i, orig_idx in enumerate(valid_indices):
            result[orig_idx] = embeddings[i].tolist()

        return result

class ZoteroEmbedder:
    def __init__(self, storage_path: str, persist_directory: str = "chroma_db"):
        """
        Initialize the ZoteroEmbedder
        
        Args:
            storage_path: Path to local Zotero storage (e.g., '~/Zotero/storage')
            persist_directory: Directory to store the Chroma database
        """
        # Get credentials from environment variables
        library_id = os.getenv('LIBRARY_ID')
        api_key = os.getenv('API_KEY')
        
        if not library_id or not api_key:
            raise ValueError("Missing LIBRARY_ID or API_KEY in environment variables")
        
        self.zot = zotero.Zotero(library_id, 'user', api_key)
        
        # Expand and verify storage path
        self.storage_path = os.path.expanduser(storage_path)
        if not os.path.exists(self.storage_path):
            raise ValueError(f"Zotero storage path not found: {self.storage_path}")
            
        print(f"Using Zotero storage path: {self.storage_path}")
        
        # Set up device for M1 Mac
        self.device = (
            "mps" 
            if torch.backends.mps.is_available() 
            else "cuda" 
            if torch.cuda.is_available() 
            else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Initialize the model with the appropriate device
        self.model_name = 'all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.model_name)
        self.model.to(self.device)
        
        # Initialize ChromaDB with custom embedding function
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = CustomEmbeddingFunction(self.model, self.device)
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="zotero_papers",
            embedding_function=self.embedding_function
        )

    def get_pdf_content(self, item) -> str:
        """
        Retrieve PDF content from local storage
        """
        try:
            # Get child attachments
            children = self.zot.children(item['key'])
            
            # Find PDF attachment
            for child in children:
                if child['data'].get('contentType') == 'application/pdf':
                    pdf_key = child['key']
                    # Check the storage folder for PDFs
                    folder_path = os.path.join(self.storage_path, pdf_key)
                    if os.path.exists(folder_path):
                        # Get all PDFs in this folder
                        pdfs = glob.glob(os.path.join(folder_path, "*.pdf"))
                        if pdfs:
                            try:
                                # Use the first PDF found
                                doc = fitz.open(pdfs[0])
                                text = ""
                                for page in doc:
                                    text += page.get_text()
                                doc.close()
                                return text
                            except Exception as e:
                                print(f"Error reading PDF for {item['data'].get('title', 'Unknown')}: {str(e)}")
                        
            return ""  # Return empty string if no PDF found
        except Exception as e:
            print(f"Error accessing children for {item['data'].get('title', 'Unknown')}: {str(e)}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or len(text.strip()) == 0:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            start = end - overlap
            
        return chunks

    def create_embeddings(self, chunk_size: int = 1000, overlap: int = 100, batch_size: int = 32):
        """
        Create embeddings for all PDFs in the Zotero library and store in ChromaDB
        """
        processed_pdfs = 0
        total_chunks = 0
        skipped_items = 0
        
        # Get total number of items
        total_items = self.zot.count_items()
        start = 0
        limit = 100  # Zotero's maximum items per request
        
        print(f"Found {total_items} total items in Zotero library")
        
        with tqdm(total=total_items) as pbar:
            while start < total_items:
                # Get items with pagination
                items = self.zot.items(start=start, limit=limit)
                
                for item in items:
                    try:
                        # Only process items that are not attachments themselves
                        if item['data'].get('itemType') != 'attachment':
                            # Get PDF content
                            text = self.get_pdf_content(item)
                            
                            if text and len(text.strip()) > 0:  # Check for non-empty text
                                # Split into chunks
                                chunks = self.chunk_text(text, chunk_size, overlap)
                                
                                if chunks:  # Only process if we have valid chunks
                                    total_chunks += len(chunks)
                                    processed_pdfs += 1
                                    
                                    # Convert authors to string format
                                    authors = item['data'].get('creators', [])
                                    author_strings = []
                                    for author in authors:
                                        if 'firstName' in author and 'lastName' in author:
                                            author_strings.append(f"{author['firstName']} {author['lastName']}")
                                        elif 'lastName' in author:
                                            author_strings.append(author['lastName'])
                                    
                                    # Prepare metadata with string-based authors
                                    metadata = {
                                        'title': item['data'].get('title', ''),
                                        'authors': '; '.join(author_strings),
                                        'year': item['data'].get('date', ''),
                                        'doi': item['data'].get('DOI', ''),
                                        'tags': ', '.join([tag.get('tag', '') for tag in item['data'].get('tags', [])]),
                                        'zotero_key': item['key']
                                    }
                                    
                                    # Process chunks in batches
                                    for i in range(0, len(chunks), batch_size):
                                        batch_chunks = chunks[i:i + batch_size]
                                        if batch_chunks:  # Only process non-empty batches
                                            batch_ids = [str(uuid.uuid4()) for _ in batch_chunks]
                                            batch_metadatas = [{
                                                **metadata,
                                                'chunk_index': idx + i,
                                                'total_chunks': len(chunks)
                                            } for idx in range(len(batch_chunks))]
                                            
                                            # Add batch to ChromaDB
                                            self.collection.add(
                                                documents=batch_chunks,
                                                ids=batch_ids,
                                                metadatas=batch_metadatas
                                            )
                                else:
                                    skipped_items += 1
                            else:
                                skipped_items += 1
                                
                    except Exception as e:
                        skipped_items += 1
                        print(f"\nError processing {item['data'].get('title', 'Unknown')}: {str(e)}")
                    
                    pbar.update(1)
                
                start += limit
        
        # Print summary statistics
        print("\nProcessing complete!")
        print(f"Successfully processed {processed_pdfs} PDFs")
        print(f"Created {total_chunks} chunks across all documents")
        print(f"Average chunks per document: {total_chunks/processed_pdfs if processed_pdfs > 0 else 0:.2f}")
        print(f"Skipped {skipped_items} items")
        
        return {
            'processed_pdfs': processed_pdfs,
            'total_chunks': total_chunks,
            'skipped_items': skipped_items,
            'avg_chunks_per_doc': total_chunks/processed_pdfs if processed_pdfs > 0 else 0
        }

    def search(self, query: str, n_results: int = 5):
        """
        Search through embeddings using ChromaDB
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        for idx in range(len(results['documents'][0])):
            formatted_results.append({
                'chunk': results['documents'][0][idx],
                'metadata': results['metadatas'][0][idx],
                'distance': results['distances'][0][idx]
            })
            
        return formatted_results


    def get_collection_stats(self):
        """Get statistics about the embedded collection"""
        return {
            'total_chunks': self.collection.count(),
            'total_documents': len(set(m['zotero_key'] for m in self.collection.get()['metadatas']))
        }
    
    def test_pdf_access(self, limit: int = 5):
        """Test PDF access for the first few items"""
        items = self.zot.items(limit=limit)
        for item in items:
            if item['data'].get('itemType') != 'attachment':
                print(f"\nChecking: {item['data'].get('title', 'Unknown')}")
                children = self.zot.children(item['key'])
                for child in children:
                    if child['data'].get('contentType') == 'application/pdf':
                        pdf_key = child['key']
                        folder_path = os.path.join(self.storage_path, pdf_key)
                        print(f"Looking for PDFs in: {folder_path}")
                        if os.path.exists(folder_path):
                            pdfs = glob.glob(os.path.join(folder_path, "*.pdf"))
                            for pdf in pdfs:
                                print(f"Found PDF: {os.path.basename(pdf)}")
                        else:
                            print("Folder not found")