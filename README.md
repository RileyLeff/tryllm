# Riley's LLM Playground Folder

Just screwing around with LLM apis.

To use this, you'll need a local .env (make sure you don't commit it to version control!) of the format:
API_KEY=zotero_api_key_unquoted
LIBRARY_ID=zotero_library_id_unquoted
OPENAI_API_KEY=optional_chatgpt_api_key_unquoted
ANTHROPIC_API_KEY=optional_anthropic_api_key_unquoted
DEEPSEEK_API_KEY=optional_deepseek_api_key_unquoted

This project uses uv, the best python version and dependency manager.

Locally modify and run the script use_assistant.py to give it a shot. This is all scattershot programming to try stuff out, so it's pretty likely there will be some gotchas regarding hard-coded file paths, assumptions about system dependencies, etc.

Expect locally generating the embeddings to take a few minutes, especially if you have weak/no gpu.

Oh and heads up that I used the old (now deprecated) strat for connecting to openai, need to fix that. Only claude and deepseek work right now.
