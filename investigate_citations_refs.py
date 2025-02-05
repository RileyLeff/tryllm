from semanticscholar import SemanticScholar
import os
from dotenv import load_dotenv
from pprint import pprint

# Setup
load_dotenv()
api_key = os.getenv('SEMANTIC_API_KEY')
sch = SemanticScholar(api_key=api_key)

# Get a test paper
doi = "10.1111/gcb.12986"
paper = sch.get_paper(doi)
print(f"\nPaper ID: {paper.paperId}")
print(f"Reference count from metadata: {paper.referenceCount}")

# Get citations
citations = sch.get_paper_citations(paper.paperId)
print(f"\nFound {len(citations)} citations")

# Get references
references = sch.get_paper_references(paper.paperId)
print(f"\nReferences object: {references}")

# Now you can interactively inspect:
# paper
# citations
# references
# 
# Try things like:
# len(references)
# dir(references)
# vars(references[0]) if len(references) > 0