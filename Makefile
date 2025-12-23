.PHONY: build up down logs shell test clean

# Build Docker image
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 10
	@echo "Services started! API available at http://localhost:8000"

# Stop all services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f voice-platform

# Shell into container
shell:
	docker-compose exec voice-platform bash

# Run tests
test:
	docker-compose exec voice-platform python scripts/test_websocket.py

# Health check
health:
	@curl -s http://localhost:8000/health | python -m json.tool

# Clean up
clean:
	docker-compose down -v --rmi local
	docker system prune -f

# Development: run locally
dev:
	PYTHONPATH=src python scripts/run_server.py
