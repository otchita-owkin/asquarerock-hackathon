.PHONY: clean clean-test clean-docs clean-pyc clean-build docs help
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
	poetry install --with=linting

clean: clean-build clean-pyc clean-test clean-docs ## remove all build, test, coverage and Python artifacts

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

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -f coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr prof/

clean-docs: ## remove docs artifacts
	rm -fr docs/_build
	rm -fr docs/api

format: ## format code by sorting imports with black
	black a2rock tests

lint: ## check style with flake8
	flake8 a2rock tests
	pylint a2rock tests

typing: ## check static typing using mypy
	mypy a2rock

pre-commit-checks: ## Run pre-commit checks on all files
	pre-commit run --hook-stage manual --all-files --show-diff-on-failure

lint-all: pre-commit-checks lint typing ## Run all linting checks.

test: ## run tests quickly with the default Python
	pytest

test-docs: docs-api ## check docs using doc8
	pydocstyle a2rock
	doc8 docs
	$(MAKE) -C docs doctest

coverage: ## check code coverage quickly with the default Python
	coverage run --source a2rock -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

profile:  ## create a profile from test cases
	$(PROFILE) $(TARGET)
	$(PROFILE_VIEWER) $(PROFILE_RESULT)

docs-api:  ## generate the API documentation for Sphinx
	rm -rf docs/api
	sphinx-apidoc -e -M -o docs/api a2rock

docs: docs-api ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .
