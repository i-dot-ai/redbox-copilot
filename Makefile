
-include .env

.PHONY: app reqs

BACKUP_ENV_FILENAME = ".env.$(shell date +"%Y-%m-%d-%H:%M:%S").backup"

reqs:
	poetry install

run:
	docker compose up -d elasticsearch kibana worker minio redis core-api db django-app

stop:
	docker compose down

clean:
	docker compose down -v --rmi all --remove-orphans

build:
	docker compose build

rebuild:
	docker compose build --no-cache

test-core-api:
	poetry install --no-root --no-ansi --with api,dev,ai --without worker
	poetry run pytest core_api/tests --cov=core_api/src -v --cov-report=term-missing --cov-fail-under=45

test-redbox:
	poetry install --no-root --no-ansi --with api,dev --without ai,worker
	poetry run pytest redbox/tests --cov=redbox -v --cov-report=term-missing --cov-fail-under=80

test-worker:
	poetry install --no-root --no-ansi --with worker,dev --without ai,api
	poetry run pytest worker/tests --cov=worker -v --cov-report=term-missing --cov-fail-under=40

test-django:
	docker compose up -d --wait db minio
	docker compose run django-app venv/bin/pytest tests/ --ds redbox_app.settings -v --cov=redbox_app.redbox_core --cov-fail-under 80 -o log_cli=true

test-integration: stop
#	cp .env $(BACKUP_ENV_FILENAME)
#	cp .env.integration .env
	docker compose up -d --wait elasticsearch db worker minio core-api django-app
	poetry install --no-root --no-ansi --with dev --without ai,api,worker,docs
	sleep 10
	poetry run pytest tests/
#	cp $(BACKUP_ENV_FILENAME) .env
#	rm $(BACKUP_ENV_FILENAME)

collect-static:
	docker compose run django-app venv/bin/django-admin collectstatic --noinput

lint:
	poetry run ruff format . --check
	poetry run ruff check .

format:
	poetry run ruff format .
	poetry run ruff check . --fix

safe:
	poetry run bandit -ll -r ./redbox
	poetry run bandit -ll -r ./django_app
	poetry run mypy ./redbox --ignore-missing-imports
	poetry run mypy ./django_app --ignore-missing-imports

checktypes:
	poetry run mypy redbox worker --ignore-missing-imports --no-incremental

check-migrations:
	docker compose build django-app
	docker compose up -d --wait db minio
	docker compose run django-app venv/bin/django-admin migrate
	docker compose run django-app venv/bin/django-admin makemigrations --check

reset-db:
	docker compose down db --volumes
	docker compose up -d db

docs-serve:
	poetry run mkdocs serve

docs-build:
	poetry run mkdocs build

# Docker
AWS_REGION=eu-west-2
APP_NAME=redbox
ECR_URL=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
ECR_REPO_URL=$(ECR_URL)/$(ECR_REPO_NAME)
IMAGE=$(ECR_REPO_URL):$(IMAGE_TAG)

ECR_REPO_NAME=$(APP_NAME)
PREV_IMAGE_TAG=$$(git rev-parse HEAD~1)
IMAGE_TAG=$$(git rev-parse HEAD)

tf_build_args=-var "image_tag=$(IMAGE_TAG)"
DOCKER_SERVICES=$$(docker compose config --services | grep -v mlflow)

.PHONY: docker_login
docker_login:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_URL)

.PHONY: docker_build
docker_build: ## Build the docker container
	@cp .env.example .env
	# Fetching list of services defined in docker compose configuration
	@echo "Services to update: $(DOCKER_SERVICES)"
	# Enabling Docker BuildKit for better build performance
	export DOCKER_BUILDKIT=1
	@for service in $(DOCKER_SERVICES); do \
		if grep -A 2 "^\s*$$service:" docker-compose.yml | grep -q 'build:'; then \
			echo "Building $$service..."; \
			PREV_IMAGE="$(ECR_REPO_URL)-$$service:$(PREV_IMAGE_TAG)"; \
			echo "Pulling previous image: $$PREV_IMAGE"; \
			docker pull $$PREV_IMAGE; \
			docker compose build $$service; \
		else \
			echo "Skipping $$service uses default image"; \
		fi; \
	done


.PHONY: docker_push
docker_push:
	@echo "Services to push: $(DOCKER_SERVICES)"
	@for service in $(DOCKER_SERVICES); do \
		if grep -A 2 "^\s*$$service:" docker-compose.yml | grep -q 'build:'; then \
			echo "Pushing $$service..."; \
			ECR_REPO_SERVICE_TAG=$(ECR_REPO_URL)-$$service:$(IMAGE_TAG); \
			CURRENT_TAG=$$(grep -A 1 "^\s*$$service:" docker-compose.yml | grep 'image:' | sed 's/.*image:\s*//'); \
			echo "Tagging $$service: $$CURRENT_TAG -> $$ECR_REPO_SERVICE_TAG"; \
			docker tag $$CURRENT_TAG $$ECR_REPO_SERVICE_TAG; \
			docker push $$ECR_REPO_SERVICE_TAG; \
		else \
			echo "Skipping $$service uses default image"; \
		fi; \
	done

.PHONY: docker_update_tag
docker_update_tag:
	MANIFEST=$$(aws ecr batch-get-image --repository-name $(ECR_REPO_NAME) --image-ids imageTag=$(IMAGE_TAG) --query 'images[].imageManifest' --output text) && \
	aws ecr put-image --repository-name $(ECR_REPO_NAME) --image-tag $(tag) --image-manifest "$$MANIFEST"


# Ouputs the value that you're after - usefx	ul to get a value i.e. IMAGE_TAG out of the Makefile
.PHONY: docker_echo
docker_echo:
	echo $($(value))

CONFIG_DIR=../../../redbox-copilot-infra-config
TF_BACKEND_CONFIG=$(CONFIG_DIR)/backend.hcl

tf_new_workspace:
	terraform -chdir=./infrastructure/aws workspace new $(env)

tf_set_workspace:
	terraform -chdir=./infrastructure/aws workspace select $(env)

tf_set_or_create_workspace:
	make tf_set_workspace || make tf_new_workspace

.PHONY: tf_init
tf_init: ## Initialise terraform
	terraform -chdir=./infrastructure/aws init -backend-config=$(TF_BACKEND_CONFIG)

.PHONY: tf_plan
tf_plan: ## Plan terraform
	make tf_set_workspace && \
	terraform -chdir=./infrastructure/aws plan -var-file=$(CONFIG_DIR)/${env}-input-params.tfvars ${tf_build_args}

.PHONY: tf_apply
tf_apply: ## Apply terraform
	make tf_set_workspace && \
	terraform -chdir=./infrastructure/aws apply -var-file=$(CONFIG_DIR)/${env}-input-params.tfvars ${tf_build_args}

.PHONY: tf_auto_deploy
tf_auto_deploy: ## Auto deploy terraform to specified environment with specified tag (default: dev and HEAD)
	make tf_set_workspace && \
	terraform -chdir=./infrastructure/aws apply -auto-approve -lock-timeout=300s -var-file=$(CONFIG_DIR)/$(env)-input-params.tfvars -var=image_tag=$(IMAGE_TAG) ${tf_build_args}

.PHONY: tf_init_universal
tf_init_universal: ## Initialise terraform
	terraform -chdir=./infrastructure/aws/universal init -backend-config=../$(TF_BACKEND_CONFIG)

.PHONY: tf_apply_universal
tf_apply_universal: ## Apply terraform
	terraform -chdir=./infrastructure/aws workspace select prod && \
	terraform -chdir=./infrastructure/aws/universal apply -var-file=../$(CONFIG_DIR)/prod-input-params.tfvars

.PHONY: tf_auto_apply
tf_auto_apply: ## Auto apply terraform
	make tf_set_workspace && \
	terraform -chdir=./infrastructure/aws apply -auto-approve -var-file=$(CONFIG_DIR)/${env}-input-params.tfvars ${tf_build_args}

.PHONY: tf_destroy
tf_destroy: ## Destroy terraform
	make tf_set_workspace && \
	terraform -chdir=./infrastructure/aws destroy -var-file=$(CONFIG_DIR)/${env}-input-params.tfvars ${tf_build_args}

# Release commands to deploy your app to AWS
.PHONY: release
release: ## Deploy app
	chmod +x ./infrastructure/scripts/release.sh && ./infrastructure/scripts/release.sh $(env)

# Runs the only the necessary backend for evaluation BUCKET_NAME
.PHONY: eval_backend
eval_backend:
	docker compose up core-api -d --build
	docker exec -it $$(docker ps -q --filter "name=minio") mc mb data/$${BUCKET_NAME}
