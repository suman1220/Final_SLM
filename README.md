# Medical AI Chatbot - Diabetes Care

A production-ready medical chatbot powered by a local LLM, specialized in diabetes management and care.

## Features

- **Two Operating Modes**
  - **Model ONLY**: Direct LLM responses (1-2 line answers)
  - **Model + Guardrail**: Validated responses with medical policy enforcement

- **Guardrails System**
  - Restricts responses to diabetes/healthcare topics only
  - Enforces response length limits (2-4 sentences)
  - Prevents greetings and off-topic discussions
  - Standard refusal for non-medical queries

- **Production Features**
  - Session-based conversation memory (last 20 Q&A pairs)
  - Response caching (30-minute TTL)
  - Real-time streaming responses
  - CORS support for cross-origin requests
  - Health check endpoint for monitoring
  - First Token Time (TTFT) metrics displayed in UI

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and set your model path:

```
MODEL_PATH=model.gguf
LLM_THREADS=6
MAX_TOKENS=180
```

### 3. Run the Application

```bash
python app.py
```

The server will start on `http://0.0.0.0:8001`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `model.gguf` | Path to GGUF model file |
| `LLM_THREADS` | `6` | Number of CPU threads |
| `LLM_BATCH` | `256` | Batch size for inference |
| `MAX_TOKENS` | `180` | Maximum tokens per response |
| `FLASK_ENV` | `production` | Environment (production/development) |
| `PORT` | `8001` | Server port |

## API Endpoints

### Health Check
```
GET /api/health
```
Returns server health status and model load state.

### Get Configuration
```
GET /api/config
```
Returns current model configuration.

### Chat Stream
```
POST /chat_stream
Content-Type: application/json

{
  "question": "What is diabetes?",
  "mode": "guardrail"
}
```
Streams response via Server-Sent Events (SSE).

### Clear Memory
```
POST /api/clear_memory
X-Session-Id: <session-uuid>
```
Clears conversation memory for the session.

## Project Structure

```
SLM_UI/
├── app.py                    # Main Flask application (simplified, production-ready)
├── templates/
│   └── index.html            # Web UI with TTFT metrics
├── llm/
│   ├── guardrails_medical.py # Medical policy validation
│   ├── guardrails_security.py # Security filters (prompt injection, crisis detection)
│   ├── main.py               # Standalone LLM runner with guardrails
│   ├── main_only_model.py    # Simple model-only runner
│   └── ragModel.py           # RAG-based model runner
├── static/                   # Static assets (CSS, JS - if needed)
├── model.gguf                # LLM model file (not in git)
├── .env                      # Environment configuration
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container definition
└── docker-compose.yml        # Docker Compose orchestration
```

## UI Features

The web interface displays real-time metrics:

- **Retrieval Time**: Time to prepare the prompt
- **TTFT (Time to First Token)**: How quickly the first token is generated
- **Total Time**: Complete response generation time

These metrics help monitor model performance and user experience.

## Guardrails Policy

The medical guardrails enforce:

1. **Topic Restriction**: Only diabetes/healthcare questions
2. **Length Control**: Maximum 4 non-empty lines
3. **No Greetings**: Responses start directly with medical information
4. **Keyword Validation**: Must contain diabetes-related terms
5. **Standard Refusal**: "I'm your virtual healthcare professional, and I can only assist with diabetes and health-related questions."

## Security Considerations

- Model runs locally (no external API calls)
- CORS configured (update `CORS_ORIGINS` in `.env`)
- Input validation for question length
- Session-based memory isolation
- No sensitive data logging

## Performance Optimization

- **Caching**: 30-minute response cache for identical questions
- **Streaming**: Real-time token streaming for faster perceived response
- **Thread Control**: Adjustable CPU thread allocation
- **Memory Management**: Fixed-size conversation history per session

## Troubleshooting

### Model Loading Fails
```
Error: Model file not found
```
**Solution**: Ensure `MODEL_PATH` in `.env` points to a valid `.gguf` file.

### CORS Errors
```
Access-Control-Allow-Origin error
```
**Solution**: Add your frontend domain to `CORS_ORIGINS` in `.env`.

### Slow Responses
**Solutions**:
- Increase `LLM_THREADS` (but don't exceed your CPU cores)
- Reduce `MAX_TOKENS` for shorter responses
- Use a smaller/quantized model

### High Memory Usage
**Solutions**:
- Reduce `MEMORY_MAXLEN` (default: 20)
- Decrease `CACHE_TTL_SECONDS` (default: 1800)
- Use a smaller model file

## Development

To run in development mode:

```bash
export FLASK_ENV=development
export FLASK_DEBUG=True
python app.py
```

## Docker Deployment (Recommended)

### Quick Start with Docker

1. **Ensure you have the model file**:
```bash
# Place your model.gguf in the project root directory
ls -lh model.gguf
```

2. **Build and run with Docker Compose**:
```bash
docker compose up -d
# OR if using older docker-compose:
# docker-compose up -d
```

3. **Access the application**:
```
http://localhost:8001
```

4. **View logs**:
```bash
docker compose logs -f
```

5. **Stop the application**:
```bash
docker compose down
```

### Manual Docker Build

```bash
# Build the image
docker build -t diabetes-chatbot .

# Run the container
docker run -d \
  -p 8001:8001 \
  -v $(pwd)/model.gguf:/app/model.gguf:ro \
  -e LLM_THREADS=6 \
  -e MAX_TOKENS=150 \
  --name diabetes-chatbot \
  diabetes-chatbot
```

### Docker Environment Variables

Override defaults in `docker-compose.yml` or pass via `-e`:

```yaml
environment:
  - MODEL_PATH=model.gguf
  - LLM_THREADS=6          # Adjust based on your CPU
  - LLM_BATCH=256
  - MAX_TOKENS=150
```

## Production Deployment (Without Docker)

1. Set environment variables:
```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
export SECRET_KEY=your-secret-key-here
```

2. Use a production server (gunicorn):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8001 app:app
```

3. Set up reverse proxy (nginx):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        chunked_transfer_encoding off;
    }
}
```

## Monitoring

Check application health:
```bash
curl http://localhost:8001/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## License

Proprietary - Internal Use Only

## Support

For issues or questions, contact the development team.
