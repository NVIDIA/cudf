# cuDF Development Containers

This directory contains [devcontainer configurations](https://containers.dev/implementors/json_reference/) for using VSCode to [develop in a container](https://code.visualstudio.com/docs/devcontainers/containers) via the `Remote Containers` [extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) or [GitHub Codespaces](https://github.com/codespaces).

This container is a turnkey development environment for building and testing the cuDF C++ and Python libraries.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Host bind mounts](#host-bind-mounts)
* [Launch a Dev Container](#launch-a-dev-container)

## Prerequisites

* [VSCode](https://code.visualstudio.com/download)
* [VSCode Remote Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## Host bind mounts

By default, the following directories are bind-mounted into the devcontainer:

* `${repo}:/home/coder/cudf`
* `${repo}/../.aws:/home/coder/.aws`
* `${repo}/../.local:/home/coder/.local`
* `${repo}/../.cache:/home/coder/.cache`
* `${repo}/../.conda:/home/coder/.conda`
* `${repo}/../.config:/home/coder/.config`

This ensures caches, configurations, dependencies, and your commits are persisted on the host across container runs.

## Launch a Dev Container

To launch a devcontainer from VSCode, open the cuDF repo and select the "Reopen in Container" button in the bottom right:<br/><img src="https://user-images.githubusercontent.com/178183/221771999-97ab29d5-e718-4e5f-b32f-2cdd51bba25c.png"/>

Alternatively, open the VSCode command palette (typically `cmd/ctrl + shift + P`) and run the "Rebuild and Reopen in Container" command.
