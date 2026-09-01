# Install Python 3.10+ needed by LangGraph/LangChain/ OCI Cloud Shell has 3.9 per default. 
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
