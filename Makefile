SHELL=/bin/bash
ACTIVATE=source activate pyjanitor-dev

release:
	rm -f dist/*
	python setup.py sdist bdist_wheel
	twine upload dist/*

# Note to self:
# makefile has not been fully tested.
# DO NOT COMMIT until testing is done.
#
# ALSO, remove this comment once it's tested!!!!!!!!!!!

.PHONY: format test lint docs isort check style notebooks install

format:
	@echo "Applying Black Python code formatting..."
	pre-commit run black --all-files

test:
	@echo "Running test suite..."
	pytest -v -n auto --color=yes

lint:
	@echo "Checking code formatting..."
	pre-commit run flake8 --all-files

docs:
	@echo "Building documentation..."
	mkdocs build

isort:
	@echo "Sorting imports..."
	isort --check-only --use-parentheses --trailing-comma --multi-line 3 --line-length 79 .

check: test docs notebooks isort format lint
	@echo "checks complete"

style: isort format
	@echo "styling complete"

install:
	@echo "Creating Conda environment..."
	conda env create -f environment-dev.yml

	@echo "Installing PyJanitor in development mode..."
	$(ACTIVATE) && python setup.py develop

	@echo "Registering current virtual environment as a Jupyter Python kernel..."
	$(ACTIVATE) && python -m ipykernel install --user --name pyjanitor-dev --display-name "PyJanitor development"

	@echo "Installing pre-commit hooks"
	$(ACTIVATE) && pre-commit install

compile-requirements:
	@echo "pip-compiling requirements files..."
	find .requirements -type f -name '*.in' | xargs -I {} sh -c\
		'echo "compiling" {} && pip-compile {} --upgrade -q'
