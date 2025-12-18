.PHONY: install dev lint fmt test run

install:
	python -m pip install -U pip
	pip install -r requirements.txt

dev:
	python -m pip install -U pip
	pip install -r requirements-dev.txt
	pip install -e .

lint:
	ruff check .

fmt:
	black .

test:
	pytest -q

run:
	python -m omr_marker input.pdf output.pdf --dpi 400
