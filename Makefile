COMPOSE_FILE = docker-compose.yml
PROJECT_NAME = nids_environment
SERVICE = web

.PHONY: full_build build up down logs restart clean

full_build:
	@echo "Building base_component image using Docker..."
	docker build -t base_component:latest base_component
	@echo "Building other Docker images with docker-compose..."
	docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) build

build: 
	@echo "Building Docker images..."
	docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) build

up:
	@echo "Starting services..."
	docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) up -d

down:
	@echo "Stopping services..."
	docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down

logs:
	@echo "Showing logs..."
	docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) logs -f

restart: down up

full_clean:
	@echo "Stopping and cleaning up all containers, volumes, networks, and images..."
	docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down --volumes --remove-orphans
	docker rmi base_component:latest || true

clean:
	@echo "Cleaning up containers, volumes, and images..."
	docker compose -f $(COMPOSE_FILE) -p $(PROJECT_NAME) down --volumes --rmi all
