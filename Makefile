release:
	rm dist/*
	python setup.py sdist bdist_wheel
	twine upload dist/*

check:
	black -l 79 .
	pytest
	pycodestyle .
	cd docs && make html