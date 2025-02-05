from project import ZoteroEmbedder
from research_assistant import ResearchAssistant
import os
from dotenv import load_dotenv
from llm_providers import MODEL_CONFIGS

load_dotenv()

def list_available_models():
    print("\nAvailable models:")
    for name, config in MODEL_CONFIGS.items():
        print(f"- {name}: {config['description']}")

# List available models
list_available_models()

# Initialize the embedder
storage_path = os.path.expanduser("~/Zotero/storage")
embedder = ZoteroEmbedder(
    storage_path=storage_path,
    persist_directory='chroma_db'
)

# Choose your model
model_name = "claude-sonnet"  # or "gpt-4", "deepseek-r1", etc.
assistant = ResearchAssistant(embedder, model_name=model_name)

# Ask a research question
question = "Are coastal forests or marshes more productive ecosystems?"
answer = assistant.query(question)
print(answer)