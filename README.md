[![Integration Tests](https://github.com/i-dot-ai/redbox-copilot/actions/workflows/integration.yml/badge.svg?branch=main)](https://github.com/i-dot-ai/redbox-copilot/actions/workflows/integration.yml?query=branch%3Amain)

# 📮 Redbox

> [!IMPORTANT]
> Incubation Project: This project is an incubation project; as such, we DON’T recommend using it in any critical use case. This project is in active development and a work in progress. This project may one day Graduate, in which case this disclaimer will be removed.

> [!NOTE]
> The original streamlit-app has moved to its own repository https://github.com/i-dot-ai/redbox-copilot-streamlit.

Redbox is a retrieval augmented generation (RAG) app that uses GenAI to chat with and summarise civil service documents. It's designed to handle a variety of administrative sources, such as letters, briefings, minutes, and speech transcripts.

- **Better retrieval**. Redbox increases organisational memory by indexing documents
- **Faster, accurate summarisation**. Redbox can summarise reports read months ago, supplement them with current work, and produce a first draft that lets civil servants focus on what they do best


https://github.com/i-dot-ai/redbox-copilot/assets/8233643/e7984242-1403-4c93-9e68-03b3f065b38d


# Setup

Please refer to the [DEVELOPER_SETUP.md](./docs/DEVELOPER_SETUP.md) for detailed instructions on setting up the project.

# Codespace
For a quick start, you can use GitHub Codespaces to run the project in a cloud-based development environment. Click the button below to open the project in a new Codespace.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/i-dot-ai/redbox-copilot?quickstart=1)

# Development

You will need to install `poppler` and `tesseract` to run the `worker`
- `brew install poppler`
- `brew install tesseract`

- Download and install [pre-commit](https://pre-commit.com) to benefit from pre-commit hooks
  - `pip install pre-commit`
  - `pre-commit install`

# Testing
- Unit tests and QA run in CI
- At this time integration test(s) take 10+ mins to run so are triggered manually in CI
- Run `make help` to see all the available build activities.

# Dependencies

This project uses a microservice architecture.

Each microservice runs in its own container defined by a `Dockerfile`.

For every microservice that we have written in python we define its dependencies using https://python-poetry.org/.

This means that our project is structured approximately like this:

```txt
redbox-copilot/
├── django_app
│  ├── app/
│  ├── static/
│  ├── tests/
│  ├── manage.py
│  ├── pyproject.toml
│  └── Dockerfile
├── worker
│  ├── src/
│  │  └── app.py
│  ├── tests/
│  └── Dockerfile
├── core-api
│  ├── src/
│  │  └── app.py
│  ├── tests/
│  └── Dockerfile
├── redbox-core/
│  ├── redbox
│  │  ├── models/
│  │  └── storage/
│  ├── tests/
│  ├── pyproject.toml
│  └── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
├── Makefile
└── README.md
```

# Contributing

We welcome contributions to this project. Please see the [CONTRIBUTING.md](./CONTRIBUTING.md) file for more information.

# License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

# Security

> [!IMPORTANT]
> The core-api is the http-gateway to the backend. Currently, this is unsecured, you should only run this on
> a private network. 

However:
* We have taken care to ensure that the backend is as stateless as possible, i.e. it only stores text chunks and 
  embeddings. All data is associated with a user, and a user can access their own data. 
* The only user data stored is the user-uuid, and no chat history is stored.
* We are considering making the core-api secure. To this end the user-uuid is passed to the core-api as a JWT.
  Currently no attempt is made to verify the JWT, but in the future we may do so, e.g. via Cognito or similar

You can generate your JWT using the following snippet. Note that you whilst you can use a more secure key than an
empty string this is currently not verified.  

```python
from jose import jwt
import requests

my_uuid = "a93a8f40-f261-4f12-869a-2cea3f3f0d71"
token = jwt.encode({"user_uuid": my_uuid}, key="")

requests.get(..., headers={"Authorization": f"Bearer {token}"})
```

You can find a link to a notebook on how to generate a JWT in the [here](./notebooks/token_generation.ipynb).

If you discover a security vulnerability within this project, please follow our [Security Policy](./SECURITY.md).

## Troubleshooting

#### Error: Elasticsearch 137

```commandline
ERROR: Elasticsearch exited unexpectedly, with exit code 137
```
This is caused by Elasticsearch not having enough memory.

Increase total memory available to 8gb.

```commandline
colima down
colima start --memory 8
```

#### Error: Docker... no space left on device

```commandline
docker: /var/lib/... no space left on device
```

This is caused by your own laptop being too full to create a new image.

Clear out old docker artefacts:

```commandline
docker system prune --all --force
```

### Frontend


#### CSS

We depend on `govuk-frontend` for GOV.UK Design System styles.

```
npm install
```

Once this has been done, `django-compressor` should work automatically to
compile the govuk-frontend SCSS on the first request and any subsequent request
after the SCSS has changed. In the meantime it will read from `frontend/CACHE`,
which is `.gitignore`d.

When we get to production, we can prepopulate `frontend/CACHE` using `manage.py
compress` before building our container, which will mean that every request
will be served from the cache.

`django-compressor` also takes care of fingerprinting and setting cache headers
for our CSS so it can be cached.

#### Fonts and images

The govuk assets are versioned in the `npm` package. On initial app setup you will need to run `poetry run python manage.py collectstatic` to copy them to the `frontend` folder from where `runserver` can serve them.

We’ll revisit this process when we deploy the app.



## How to deploy

checkout the `main` branch of the following repos:
* https://github.com/i-dot-ai/redbox-copilot
* https://github.com/i-dot-ai/i-ai-core-infrastructure/
* https://github.com/i-dot-ai/redbox-copilot-infra-config

If, and only if, you want to deploy something other than HEAD then replace `var.image_tag` in `infrastructure/aws/ecs.tf` with the hash of the build you want deployed.

Now run the commands below remembering to replace ENVIRONMENT with `dev`, `preprod` or `prod`

```commandline
cd redbox-copilot
make tf_init
make tf_apply env=<ENVIRONMENT>
```
