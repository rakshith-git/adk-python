# OpenMemory Sample

This sample demonstrates how to use OpenMemory as a self-hosted memory backend
for ADK agents.

## Prerequisites

- Python 3.9+ (Python 3.11+ recommended)
- Docker (for running OpenMemory)
- ADK installed with dependencies

## Setup

### 1. Start OpenMemory Server

Start OpenMemory using Docker:

```bash
docker run -p 3000:3000 cavira/openmemory
```

Or use the production network build:

```bash
docker run -p 3000:3000 cavira/openmemory:production
```

Verify it's running:

```bash
curl http://localhost:3000/health
```

### 2. Install Dependencies

Install ADK with OpenMemory support:

```bash
pip install google-adk[openmemory]
```

This installs `httpx` for making HTTP requests to the OpenMemory API.

### 3. Configure Environment Variables

Create a `.env` file in this directory (optional):

```bash
# Required: Google API key for the agent
GOOGLE_API_KEY=your-google-api-key

# Optional: OpenMemory base URL (defaults to localhost:3000)
OPENMEMORY_BASE_URL=http://localhost:3000

# Optional: API key if your OpenMemory instance requires authentication
# OPENMEMORY_API_KEY=your-api-key
```

**Note:** API key is only needed if your OpenMemory server is configured with authentication.

## Usage

### Basic Usage

```python
from google.adk.memory import OpenMemoryService
from google.adk.runners import Runner

# Create OpenMemory service with defaults
memory_service = OpenMemoryService(
    base_url="http://localhost:3000"
)

# Use with runner
runner = Runner(
    app_name="my_app",
    agent=my_agent,
    memory_service=memory_service
)
```

### Advanced Configuration

```python
from google.adk.memory import OpenMemoryService, OpenMemoryServiceConfig

# Custom configuration
config = OpenMemoryServiceConfig(
    search_top_k=20,              # Retrieve more memories per query
    timeout=10.0,                 # Faster timeout for production
    user_content_salience=0.9,    # Higher importance for user messages
    model_content_salience=0.75,   # Medium importance for model responses
    enable_metadata_tags=True     # Use tags for filtering
)

memory_service = OpenMemoryService(
    base_url="http://localhost:3000",
    api_key="your-api-key",
    config=config
)
```

## Running the Sample

```bash
cd contributing/samples/open_memory
python main.py
```

## Expected Output

```
----Session to create memory: <session-id> ----------------------
** User says: {'role': 'user', 'parts': [{'text': 'Hi'}]}
** model: Hello! How can I help you today?
...
Saving session to memory service...
-------------------------------------------------------------------
----Session to use memory: <session-id> ----------------------
** User says: {'role': 'user', 'parts': [{'text': 'What do I like to do?'}]}
** model: You like badminton.
...
-------------------------------------------------------------------
```

## Configuration Options

### OpenMemoryServiceConfig

- `search_top_k` (int, default: 10): Maximum memories to retrieve per search
- `timeout` (float, default: 30.0): HTTP request timeout in seconds
- `user_content_salience` (float, default: 0.8): Importance for user messages
- `model_content_salience` (float, default: 0.7): Importance for model responses
- `default_salience` (float, default: 0.6): Fallback importance value
- `enable_metadata_tags` (bool, default: True): Include session/app tags

## Features

OpenMemory provides:

- **Multi-sector embeddings**: Factual, emotional, temporal, relational memory
- **Graceful decay curves**: Automatic reinforcement keeps relevant context sharp
- **Self-hosted**: Full data ownership, no vendor lock-in
- **High performance**: 2-3× faster than hosted alternatives
- **Cost-effective**: 6-10× cheaper than SaaS memory APIs

## Learn More

- [OpenMemory Documentation](https://openmemory.cavira.app/)
- [OpenMemory API Reference](https://openmemory.cavira.app/docs/api/add-memory)
- [ADK Memory Documentation](https://google.github.io/adk-docs)

