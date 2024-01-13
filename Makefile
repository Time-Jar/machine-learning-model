
training-deps:
	pip install -r training/requirements.txt

server-run:
	python -m server.app

server-build:
	sudo docker build -t supabase-tensorflow-api:latest .

server-deploy:
	sudo docker compose up