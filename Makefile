
training-deps:
	pip install -r training/requirements.txt

server-build:
	sudo docker build -t server-tensorflow-api:latest .

server-deploy:
	sudo docker-compose up