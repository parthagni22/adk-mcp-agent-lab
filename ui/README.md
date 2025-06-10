# Host Agent UI Implementations

This directory contains two different UI implementations for the Host Agent, showcasing different architectural approaches.

## Architecture Comparison

### 1. Embedded UI Architecture (`app.py`)

- **Pattern**: Direct ADK Runner integration
- **Communication**: In-process function calls
- **State Management**: `@st.cache_resource` for persistent ADK Runner
- **Event Processing**: Real-time streaming of ADK events
- **Memory**: Agent context preserved via persistent session

### 2. A2A (Agent-to-Agent) Architecture (`a2a_app.py`)

- **Pattern**: HTTP-based service communication
- **Communication**: A2A protocol over HTTP
- **State Management**: Polling-based task status checking
- **Event Processing**: Request/response with task ID tracking
- **Memory**: Agent context managed by remote service

## Running the UIs

### Prerequisites

Ensure you have the required services running:

```bash
# Start child agents (ElevenLabs and Notion)
python -m elevenlabs_agent --port 8003
python -m notion_agent --port 8004

# For A2A UI, also start the Host Agent A2A service
python -m host_agent --port 8001
```

### Embedded UI (app.py)

```bash
# From the ui directory
streamlit run app.py --server.port 8501
```

- **URL**: http://localhost:8501
- **Icon**: 🤖 Host Agent Assistant
- **Memory**: Direct ADK Runner with conversation context

### A2A UI (a2a_app.py)

```bash
# From the ui directory
streamlit run a2a_app.py --server.port 8502
```

- **URL**: http://localhost:8502
- **Icon**: 🌐 Host Agent Assistant (A2A)
- **Memory**: Via A2A service session management

## Side-by-Side Comparison

You can run both UIs simultaneously on different ports to compare:

1. **Terminal 1**: `streamlit run ui/app.py --server.port 8501`
2. **Terminal 2**: `streamlit run ui/a2a_app.py --server.port 8502`
3. **Terminal 3**: `python -m host_agent --port 8001` (for A2A UI)

Visit both URLs and test the same queries to see the differences in:

- Response times
- Error handling
- Session management
- Tool visibility

## Key Differences

| Feature             | Embedded UI      | A2A UI            |
| ------------------- | ---------------- | ----------------- |
| **Startup**         | Immediate        | Requires service  |
| **Latency**         | Low (in-process) | Higher (HTTP)     |
| **Scaling**         | Single process   | Multi-service     |
| **Debugging**       | Direct events    | Task polling      |
| **Deployment**      | Monolithic       | Microservices     |
| **Fault Tolerance** | Shared fate      | Service isolation |

## Test Scenarios

Try these queries in both UIs to compare behavior:

1. **Simple Notion Search**: "Search my Notion workspace for 'sermon notes'"
2. **Database Query**: "Count how many sermon notes are in the database"
3. **Full Orchestration**: "Find my sermon notes and read the titles of the first two entries aloud"
4. **TTS Only**: "Convert this text to speech: 'Hello from the Host Agent'"

## Troubleshooting

### Embedded UI Issues

- Check if child agents are running on ports 8003, 8004
- Verify environment variables in `.env`
- Look for ADK Runner initialization logs

### A2A UI Issues

- Ensure Host Agent A2A service is running on port 8001
- Check `HOST_AGENT_A2A_URL` environment variable
- Verify A2A client connectivity with test_client.py

### Common Issues

- **Port conflicts**: Change port numbers in commands
- **Missing environment**: Check `.env` file has API keys
- **Service discovery**: Verify agent URLs are correct
