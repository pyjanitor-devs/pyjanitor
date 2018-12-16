release:
	rm dist/*
	python setup.py sdist bdist_wheel
	twine upload dist/*

# Note to self:
# makefile has not been fully tested.
# DO NOT COMMIT until testing is done.
#
# ALSO, remove this comment once it's tested!!!!!!!!!!!

format:
	black -l 79 .

test:
	pytest

lint:
	pycodestyle .

docs:
	cd docs && make html

isort:
	isort -r . -y -up -tc

check: test lint docs isort format
	echo "checks complete"