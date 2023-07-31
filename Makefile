quality_checks:
	isort .
	black .

run_training: quality_checks
	bash run.sh