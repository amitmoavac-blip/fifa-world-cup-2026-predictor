.PHONY: install ingest fit predict backtest report test ui serve

install:
	pip install -e ".[dev]"

ingest:
	python -m wc26.cli ingest

fit:
	python -m wc26.cli fit --asof $(or $(DATE),today)

predict:
	python -m wc26.cli predict --date $(or $(DATE),today)

backtest:
	python -m wc26.cli backtest

report:
	python -m wc26.cli report

test:
	python -m pytest -q

ui:
	python -m wc26.cli build-ui-data

serve:
	python -m wc26.cli serve --rebuild --port $(or $(PORT),8000)
