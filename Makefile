start-vm:
	gcloud compute instances start em-pipeline --zone=us-east4-b

stop-vm:
	gcloud compute instances stop em-pipeline --zone=us-east4-b

run-local:
	poetry run python gdelt_ml_pipeline/main.py
