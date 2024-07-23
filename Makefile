makefile_name := $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))

-include .env

default: help

.PHONY: reqs
reqs:
	poetry install

.PHONY: run
run: stop
	docker compose up -d --wait django-app

.PHONY: stop
stop: ## Stop all containers
	docker compose down

.PHONY: prune
prune: stop
	docker system prune --all --force

.PHONY: clean
clean:
	docker compose down -v --rmi all --remove-orphans

.PHONY: build
build:
	docker compose build

.PHONY: rebuild
rebuild: stop prune ## Rebuild all images
	docker compose build --no-cache

.PHONY: test-core-api
test-core-api: ## Test core-api
	cp .env.test core-api/.env
	cd core-api && poetry install --with dev && poetry run python -m pytest -m "not ai" --cov=core_api -v --cov-report=term-missing --cov-fail-under=80

.PHONY: test-ai
test-ai: ## Test code with live LLM
	cp .env.test core-api/.env
	cd core-api && poetry install --with dev && poetry run python -m pytest -m "ai" --cov=core_api -v --cov-report=term-missing --cov-fail-under=80

.PHONY: test-redbox
test-redbox: ## Test redbox
	cp .env.test redbox-core/.env
	cd redbox-core && poetry install && poetry run pytest --cov=redbox -v --cov-report=term-missing --cov-fail-under=60

.PHONY: test-worker
test-worker: ## Test worker
	cp .env.test worker/.env
	cd worker && poetry install && poetry run pytest --cov=worker -v --cov-report=term-missing --cov-fail-under=80

.PHONY: test-django
test-django: stop ## Test django-app
	docker compose up -d --wait db minio
	docker compose run --no-deps django-app venv/bin/pytest tests/ --ds redbox_app.settings -v --cov=redbox_app.redbox_core --cov-fail-under 85 -o log_cli=true

.PHONY: build-django-static
build-django-static: ## Build django-app static files
	cd django-app/frontend/ && npm install && npm run build
	cd django-app/ && poetry run python manage.py collectstatic --noinput

.PHONY: test-integration
test-integration: rebuild run test-integration-without-build ## Run all integration tests

.PHONY: test-integration-without-build
test-integration-without-build : ## Run all integration tests without rebuilding
	poetry install --no-root --no-ansi --with dev --without docs
	poetry run pytest tests/

.PHONY: collect-static
collect-static:
	docker compose run django-app venv/bin/django-admin collectstatic --noinput

.PHONY: lint
lint:  ## Check code formatting & linting
	poetry run ruff format . --check
	poetry run ruff check .

.PHONY: format
format:  ## Format and fix code
	poetry run ruff format .
	poetry run ruff check . --fix

.PHONY: safe
safe:  ##
	poetry run bandit -ll -r ./redbox
	poetry run bandit -ll -r ./django-app
	poetry run mypy ./redbox --ignore-missing-imports
	poetry run mypy ./django-app --ignore-missing-imports

.PHONY: checktypes
checktypes:  ## Check types in redbox and worker
	poetry install --with dev --without docs --no-root
	poetry run mypy redbox-core --ignore-missing-imports
	poetry run mypy worker --ignore-missing-imports

.PHONY: check-migrations
check-migrations: stop  ## Check types in redbox and worker
	docker compose build django-app
	docker compose up -d --wait db minio
	docker compose run --no-deps django-app venv/bin/django-admin migrate
	docker compose run --no-deps django-app venv/bin/django-admin makemigrations --check

.PHONY: reset-db
reset-db:  ## Reset Django database
	docker compose down db --volumes
	docker compose up -d db

.PHONY: reset-elastic
reset-elastic:  ## Reset Django database
	docker compose down elasticsearch
	rm -rf data/elastic/*
	docker compose up -d elasticsearch --wait

.PHONY: docs-serve
docs-serve:  ## Build and serve documentation
	poetry run mkdocs serve

.PHONY: docs-build
docs-build:  ## Build documentation
	poetry run mkdocs build

# Docker
AWS_REGION=eu-west-2
APP_NAME=redbox

ECR_REPO_NAME=$(APP_NAME)-$(service)
ECR_URL=$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
ECR_REPO_URL=$(ECR_URL)/$(ECR_REPO_NAME)

IMAGE_TAG=$$(git rev-parse HEAD)
PREV_IMAGE_TAG=$$(git rev-parse HEAD~1)
MAIN_IMAGE_TAG=$$(git rev-parse origin/main)

IMAGE=$(ECR_REPO_URL):$(IMAGE_TAG)
PREV_IMAGE=$(ECR_REPO_URL):$(PREV_IMAGE_TAG)

DOCKER_CACHE_BUCKET=i-dot-ai-docker-cache
DOCKER_BUILDER_CONTAINER=$(APP_NAME)

tf_build_args=-var "image_tag=$(IMAGE_TAG)"
DOCKER_SERVICES=$$(docker compose config --services | grep -v mlflow)

AUTO_APPLY_RESOURCES = module.django-app.aws_ecs_task_definition.aws-ecs-task \
                       module.django-app.aws_ecs_service.aws-ecs-service \
                       module.django-app.data.aws_ecs_task_definition.main \
                       module.core-api.aws_ecs_task_definition.aws-ecs-task \
                       module.core-api.aws_ecs_service.aws-ecs-service \
                       module.core-api.data.aws_ecs_task_definition.main \
                       module.worker.aws_ecs_task_definition.aws-ecs-task \
                       module.worker.aws_ecs_service.aws-ecs-service \
                       module.worker.data.aws_ecs_task_definition.main \
                       aws_secretsmanager_secret.django-app-secret \
                       aws_secretsmanager_secret.worker-secret \
                       aws_secretsmanager_secret.core-api-secret

target_modules = $(foreach resource,$(AUTO_APPLY_RESOURCES),-target $(resource))

.PHONY: docker_login
docker_login:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_URL)

.PHONY: docker_build_local
docker_build_local: ## Build the docker container locally
	DOCKER_BUILDKIT=1 docker build --target=runtime -t $(IMAGE) -f $(service)/Dockerfile .

ifndef cache
	override cache = ./.build-cache

.PHONY: docker_build
docker_build: ## Pull previous container (if it exists) build the docker container
	DOCKER_BUILDKIT=1 docker buildx build --target=runtime --platform linux/amd64 --load --builder=$(DOCKER_BUILDER_CONTAINER) -t $(IMAGE) \
	--cache-to type=local,dest=$(cache) \
	--cache-from type=local,src=$(cache) -f $(service)/Dockerfile .

.PHONY: docker_push
docker_push:
	docker push $(IMAGE)

.PHONY: docker_update_tag
docker_update_tag:
	MANIFEST=$$(aws ecr batch-get-image --repository-name $(ECR_REPO_NAME) --image-ids imageTag=$(IMAGE_TAG) --query 'images[].imageManifest' --output text) ; \
	aws ecr put-image --repository-name $(ECR_REPO_NAME) --image-tag $(tag) --image-manifest "$$MANIFEST" ; \

# Ouputs the value that you're after - useful to get a value i.e. IMAGE_TAG out of the Makefile
.PHONY: docker_echo
docker_echo:
	echo $($(value))

ifeq ($(instance),postgres)
 CONFIG_DIR=../../../../redbox-copilot-infra-config
 tf_build_args=
else ifeq ($(instance),universal)
 CONFIG_DIR=../../../../redbox-copilot-infra-config
 env=prod
 tf_build_args=
else
 CONFIG_DIR=../../../redbox-copilot-infra-config
 tf_build_args=-var "image_tag=$(IMAGE_TAG)"
endif

TF_BACKEND_CONFIG=$(CONFIG_DIR)/backend.hcl

tf_new_workspace:
	terraform -chdir=./infrastructure/aws/$(instance)  workspace new $(env)

tf_set_workspace:
	terraform -chdir=./infrastructure/aws/$(instance) workspace select $(env)

tf_set_or_create_workspace:
	make tf_set_workspace || make tf_new_workspace

.PHONY: tf_init
tf_init: ## Initialise terraform
	terraform -chdir=./infrastructure/aws/$(instance) init -backend-config=$(TF_BACKEND_CONFIG) ${args} -reconfigure

.PHONY: tf_plan
tf_plan: ## Plan terraform
	make tf_set_workspace && \
	terraform -chdir=./infrastructure/aws/$(instance) plan -var-file=$(CONFIG_DIR)/${env}-input-params.tfvars ${tf_build_args} -target=module.lambda-cleanup

.PHONY: tf_apply
tf_apply: ## Apply terraform
	make tf_set_workspace && \
	terraform -chdir=./infrastructure/aws/$(instance) apply -var-file=$(CONFIG_DIR)/${env}-input-params.tfvars ${tf_build_args} ${args} -target=module.lambda-cleanup -target=module.rds -target=aws_security_group_rule.lambda_to_rds_egress -target=module.elasticache -target=module.lambda-test -target=module.lambda -target=aws_security_group.service_security_group -target=aws_security_group_rule.lambda_to_443_egress

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
	terraform -chdir=./infrastructure/aws apply -auto-approve -var-file=$(CONFIG_DIR)/${env}-input-params.tfvars ${tf_build_args} $(target_modules)

.PHONY: tf_destroy
tf_destroy: ## Destroy terraform
	make tf_set_workspace && \
	terraform -chdir=./infrastructure/aws destroy -var-file=$(CONFIG_DIR)/${env}-input-params.tfvars ${tf_build_args}

.PHONY: tf_import
tf_import:
	make tf_set_workspace && \
	terraform -chdir=./infrastructure/aws/$(instance) import ${tf_build_args} -var-file=$(CONFIG_DIR)/${env}-input-params.tfvars ${name} ${id}

# Release commands to deploy your app to AWS
.PHONY: release
release: ## Deploy app
	chmod +x ./infrastructure/aws/scripts/release.sh && ./infrastructure/aws/scripts/release.sh $(env)

.PHONY: eval_backend
eval_backend:  ## Runs the only the necessary backend for evaluation BUCKET_NAME
	docker compose up -d --wait core-api --build
	docker exec -it $$(docker ps -q --filter "name=minio") mc mb data/${BUCKET_NAME}

.PHONY: help
help: ## Show this help
	@ grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(makefile_name) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1,$$2}'
