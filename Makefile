release:
	rm dist/*
	python setup.py sdist bdist_wheel
	twine upload dist/*

# Note to self:
# makefile has not been fully tested.
# DO NOT COMMIT until testing is done.
#
# ALSO, remove this comment once it's tested!!!!!!!!!!!

.PHONY: format test lint docs isort check style notebooks

black:
	black -l 79 .

test:
	pytest

lint:
	pycodestyle . --exclude ./nbconvert_config.py

docs:
	cd docs && make html

isort:
	isort -rc . -y -up -tc

notebooks:
	jupyter nbconvert --to notebook --config nbconvert_config.py --execute --template full

check: test docs notebooks isort format lint
	echo "checks complete"

style: isort format
	echo "styling complete"
