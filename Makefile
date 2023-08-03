quality_checks:
	isort .
	black .

run_train: quality_checks
	bash run.sh
