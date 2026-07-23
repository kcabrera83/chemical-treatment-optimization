# Deployment Guide - Chemical Treatment Optimization

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5009

CMD ["python", "app.py"]
```

### Build and Run

```bash
docker build -t chemical-treatment-optimization .
docker run -p 5009:5009 chemical-treatment-optimization
```

## Docker Compose

```yaml
version: '3.8'
services:
  chemical-treatment-optimization:
    build: .
    ports:
      - "5009:5009"
    environment:
      - FLASK_ENV=production
    volumes:
      - model-data:/app/outputs

volumes:
  model-data:
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment mode | development |
| PORT | Server port | 5009 |

## Production Considerations

- Use gunicorn for production serving:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5009 app:app
  ```
- Set `debug=False` in `app.py` (already set)
- Configure reverse proxy (nginx) for SSL termination
- Set up health check monitoring on `/api/health`
- Use a process manager (systemd, supervisor) for auto-restart

## Training Pipeline

1. `python train.py` generates synthetic data and trains models
2. Models are saved to `outputs/models/`
3. `python app.py` loads models and starts the API server

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Runs on push to main
- Installs dependencies
- Runs training pipeline
- Executes API tests

---

*Elaborado por Ing. Kelvin Cabrera*
