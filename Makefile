SHELL=/bin/bash
ACTIVATE=source activate pyjanitor-dev

release:
	rm dist/*
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
	black -l 79 .

test:
	@echo "Running test suite..."
	pytest --cov-report html

lint:
	@echo "Checking code formatting..."
	pycodestyle . --exclude "./nbconvert_config.py, ./env, ./venv"

docs:
	@echo "Building documentation..."
	cd docs && make html

isort:
	@echo "Sorting imports..."
	isort -rc . -y -up -tc

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
