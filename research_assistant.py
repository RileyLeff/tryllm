from project import ZoteroEmbedder
from llm_providers import MODEL_CONFIGS
from typing import List
from dotenv import load_dotenv

load_dotenv()

class ResearchAssistant:
    def __init__(self, embedder: ZoteroEmbedder, model_name: str = "gpt-4"):
        """
        Initialize the Research Assistant
        
        Args:
            embedder: Initialized ZoteroEmbedder instance
            model_name: Name of the model to use (must be in MODEL_CONFIGS)
        """
        self.embedder = embedder
        
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_name]
        self.llm = config["provider"](**config["args"])
        print(f"Using model: {config['description']}")

    def query(self, question: str, n_chunks: int = 5) -> str:
        """Query the research assistant with a question"""
        # Get relevant chunks from the embeddings
        results = self.embedder.search(question, n_results=n_chunks)
        
        # Prepare context from the chunks
        context = []
        for result in results:
            context.append(
                f"From '{result['metadata']['title']}' by {result['metadata']['authors']} ({result['metadata']['year']}):\n"
                f"{result['chunk']}\n"
            )
        
        messages = [
            {"role": "system", "content": "You are a superstar postdoctoral researcher with expertise in academic literature."},
            {"role": "user", "content": f"""Use relevant content from the following excerpts from academic papers to answer the question below.
If you're not sure about something, say so, and specifically say what you would like more information about, citing particular references or sources you would like to see more from. Include citations in your response.

Question: {question}

Relevant excerpts:
{'-' * 80}
{'\n'.join(context)}
{'-' * 80}

Please provide a detailed answer based on these sources, including specific citations where appropriate."""}
        ]

        return self.llm.generate(messages)

    def analyze_paper(self, title: str) -> str:
        """Analyze a specific paper in your database"""
        results = self.embedder.collection.get(
            where={"title": title}
        )
        
        if not results['documents']:
            return f"Paper '{title}' not found in the database."
        
        text = "\n".join(results['documents'])
        metadata = results['metadatas'][0]

        messages = [
            {"role": "system", "content": "You are a helpful research assistant with expertise in analyzing academic papers."},
            {"role": "user", "content": f"""Analyze the following academic paper and provide a comprehensive summary:

Title: {metadata['title']}
Authors: {metadata['authors']}
Year: {metadata['year']}

Content:
{text[:8000]}

Please provide:
1. Main research questions/objectives
2. Key methodology
3. Main findings
4. Significant conclusions
5. Potential implications for the field"""}
        ]

        return self.llm.generate(messages)

    def compare_papers(self, titles: List[str]) -> str:
        """Compare multiple papers in your database"""
        papers_data = []
        for title in titles:
            results = self.embedder.collection.get(
                where={"title": title}
            )
            if results['documents']:
                papers_data.append({
                    'title': title,
                    'text': "\n".join(results['documents']),
                    'metadata': results['metadatas'][0]
                })
        
        if not papers_data:
            return "None of the specified papers were found in the database."

        prompt = "Compare and contrast the following papers:\n\n"
        for paper in papers_data:
            prompt += f"""
Title: {paper['metadata']['title']}
Authors: {paper['metadata']['authors']}
Year: {paper['metadata']['year']}

Content:
{paper['text'][:4000]}

---
"""

        messages = [
            {"role": "system", "content": "You are a helpful research assistant with expertise in analyzing and comparing academic papers."},
            {"role": "user", "content": prompt + """
Please provide:
1. Key similarities in methodology and findings
2. Major differences in approach or conclusions
3. How these papers complement or contradict each other
4. Evolution of ideas if papers are from different time periods
5. Synthesis of the combined insights from these papers"""}
        ]

        return self.llm.generate(messages)