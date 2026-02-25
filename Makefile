.PHONY: api-dev web-dev docker-build demo help mock-dev mock-demo

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

api-dev: ## Start the FastAPI dev server (requires OPENAI_API_KEY)
	cd api && uvicorn main:app --reload --port 8000

mock-dev: ## Start the API in mock mode (no API keys needed)
	cd api && MOCK_MODE=1 uvicorn main:app --reload --port 8000

web-dev: ## Start the Next.js dev server
	cd web && npm run dev

docker-build: ## Build the API Docker image
	docker build -t kado-api ./api

docker-run: ## Run the API in Docker (requires OPENAI_API_KEY env var)
	docker run -p 8000:8000 -e OPENAI_API_KEY="$$OPENAI_API_KEY" kado-api

test: ## Run backend unit tests
	cd api && python -m pytest tests/ -v

demo: ## Run end-to-end demo with a test video (set VIDEO=path/to/video.mp4)
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make demo VIDEO=path/to/your/video.mp4"; exit 1; fi
	curl -X POST http://localhost:8000/analyze \
		-F "file=@$(VIDEO)" \
		-H "Accept: application/json" | python -m json.tool

mock-demo: ## Run end-to-end demo in mock mode (any file works)
	echo "dummy" > /tmp/kado_mock.mp4 && \
	curl -s -X POST http://localhost:8000/analyze \
		-F "file=@/tmp/kado_mock.mp4" \
		-H "Accept: application/json" | python3 -m json.tool && \
	rm -f /tmp/kado_mock.mp4

