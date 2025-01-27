from semanticscholar import SemanticScholar
sch = SemanticScholar()

# sperry 2017 as example
paper = sch.get_paper("10.1111/pce.12852")  
citations = sch.get_paper_citations(paper.paperId)

# Print results
print(f"Found {len(citations)} citations for {paper.title}")
for citation in citations:
    print(f"\nTitle: {citation.title}")
    if citation.authors:
        print(f"Authors: {', '.join(author.name for author in citation.authors)}")
    if citation.year:
        print(f"Year: {citation.year}")
    if citation.doi:
        print(f"DOI: {citation.doi}")