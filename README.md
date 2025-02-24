[![test](https://github.com/ks6088ts-labs/workshop-llm-agents/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/ks6088ts-labs/workshop-llm-agents/actions/workflows/test.yaml?query=branch%3Amain)
[![docker](https://github.com/ks6088ts-labs/workshop-llm-agents/actions/workflows/docker.yaml/badge.svg?branch=main)](https://github.com/ks6088ts-labs/workshop-llm-agents/actions/workflows/docker.yaml?query=branch%3Amain)
[![docker-release](https://github.com/ks6088ts-labs/workshop-llm-agents/actions/workflows/docker-release.yaml/badge.svg)](https://github.com/ks6088ts-labs/workshop-llm-agents/actions/workflows/docker-release.yaml)
[![ghcr-release](https://github.com/ks6088ts-labs/workshop-llm-agents/actions/workflows/ghcr-release.yaml/badge.svg)](https://github.com/ks6088ts-labs/workshop-llm-agents/actions/workflows/ghcr-release.yaml)
[![docs](https://github.com/ks6088ts-labs/workshop-llm-agents/actions/workflows/github-pages.yaml/badge.svg)](https://github.com/ks6088ts-labs/workshop-llm-agents/actions/workflows/github-pages.yaml)

# workshop-llm-agents

This repository is for a workshop to build LLM agents.

## Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installation)
- [GNU Make](https://www.gnu.org/software/make/)

## Architecture

![Architecture](./docs/images/workshop-llm-agents.png)

## Development instructions

### Local development

Use Makefile to run the project locally.

```shell
# help
make

# install dependencies for development
make install-deps-dev

# run tests
make test

# run CI tests
make ci-test
```

### Docker development

```shell
# build docker image
make docker-build

# run docker container
make docker-run

# run CI tests in docker container
make ci-test-docker
```

To publish the docker image to Docker Hub, you need to set the following secrets in the repository settings.

```shell
gh secret set DOCKERHUB_USERNAME --body $DOCKERHUB_USERNAME
gh secret set DOCKERHUB_TOKEN --body $DOCKERHUB_TOKEN
```
