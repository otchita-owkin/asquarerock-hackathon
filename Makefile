.PHONY: clean clean-pyc clean-build help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

ifeq (, $(shell which snakeviz))
	PROFILE = pytest --profile-svg
	PROFILE_RESULT = prof/combined.svg
	PROFILE_VIEWER = $(BROWSER)
else
    PROFILE = pytest --profile
    PROFILE_RESULT = prof/combined.prof
	PROFILE_VIEWER = snakeviz
endif

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

config: ## configure poetry with Owkin's PyPi credentials.
	$(eval USERNAME ?= $(shell bash -c 'read -p "PyPi Username: " username; echo $$username'))
	$(eval PASSWORD ?= $(shell bash -c 'read -s -p "PyPi Password: " pwd; echo $$pwd'))
	poetry config virtualenvs.in-project true
	poetry config repositories.owkin https://pypi.owkin.com/simple/
	poetry config http-basic.owkin $(USERNAME) $(PASSWORD)

lock: ## generate a new poetry.lock file (To be done after adding new requirements to pyproject.toml)
	poetry lock

install: clean ## install all package and development dependencies to the active Python's site-packages
	pip install poetry==1.2.0b2
	pip install -e hackathon_baseline
	pip install -r hackathon_baseline/requirements.txt
	poetry install --with=linting

clean: clean-build clean-pyc ## remove all build, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -path ./.venv -prune -false -o -name '*.egg-info' -exec rm -fr {} +
	find . -path ./.venv -prune -false -o -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -path ./.venv -prune -false -o -name '*.pyc' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '*.pyo' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '*~' -exec rm -f {} +
	find . -path ./.venv -prune -false -o -name '__pycache__' -exec rm -fr {} +

format: ## format code by sorting imports with black
	black a2rock

lint: ## check style with flake8
	flake8 a2rock
	pylint a2rock

typing: ## check static typing using mypy
	mypy a2rock

pre-commit-checks: ## Run pre-commit checks on all files
	pre-commit run --hook-stage manual --all-files --show-diff-on-failure

lint-all: pre-commit-checks lint typing ## Run all linting checks.
