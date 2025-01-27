import fitz  # PyMuPDF
from collections import Counter
import re

def count_words_in_pdf(pdf_path):
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    # Extract text from all pages
    text = ""
    for page in doc:
        text += page.get_text()
    
    # Clean and split the text into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count words
    word_counts = Counter(words)
    
    # Sort by frequency, most common first
    sorted_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Print results
    print(f"Total words: {len(words)}")
    print("\nTop 20 most common words:")
    for word, count in list(sorted_counts.items())[:20]:
        print(f"{word}: {count}")
    
    return word_counts

def extract_text_to_file(pdf_path, output_path):
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    # Extract text from all pages
    text = ""
    for page in doc:
        text += page.get_text()
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Text extracted to: {output_path}")

# Usage
pdf_path = "path/to/your/pdf"  # Replace with your PDF path
output_path = "sdi12.txt"     # Replace with desired output path

extract_text_to_file(pdf_path, output_path)
  # Replace with desired output path