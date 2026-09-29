install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	ruff check src dashboard tests

train:
	python -m src.models.train --data data/raw/application_train.csv

dashboard:
	streamlit run dashboard/app.py

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

docker:
	docker compose up --build
