default:
	@make run

run:
	python main.py

web:
	pygbag main.py

web-build:
	pygbag --build main.py

init:
	@pip install -U pip; \
	pip install -e ".[dev]"

format:
	black .

lint:
	flake8 --config=../.flake8 --output-file=./coverage/flake8-report --format=default
