.PHONY: app health benchmark docker-build docker-run

app:
	streamlit run apps/governance_app.py

health:
	python scripts/healthcheck.py

benchmark:
	python scripts/benchmark_cli.py samples --out logs/metrics.json

docker-build:
	docker build -t idp-app:latest .

docker-run:
	docker run --rm -p 8501:8501 -e IDP_OCR_FAST=1 idp-app:latest

.PHONY: watch
watch:
	python scripts/auto_watch.py samples --interval 10 --audit
