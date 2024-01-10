
training-deps:
	pip install -r training/requirements.txt

server-run:
	python server/app.py

server-build:
	sudo docker build -t supabase-tensorflow-api:latest .

server-deploy:
	sudo docker-compose up