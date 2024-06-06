help:
	@echo "Makefile help: (Tested on Linunx)"
	@echo "* install	to install the requirements into your current virtaul env"
	@echo "* test		to run the tests"
	@echo "* check 		to run the code style checker"

install:
	python -m pip install -r requirements.txt
	git clone https://github.com/automl/PFNs.git
	cd PFNs; pip install -e .; cd -

test:
	python -m pytest tests

check:
	pycodestyle --max-line-length=120 src
	@echo "All good!"

.PHONY: install test check all clean-tex tex help
.DEFAULT_GOAL := help
