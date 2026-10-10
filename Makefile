.PHONY: setup demo verify-demo test verify

setup:
	python -m pip install -r requirements.txt

demo:
	python demo/run_demo.py

verify-demo:
	python demo/verify_demo.py

test:
	python -m pytest -q

verify:
	python submission/verify_submission.py
	python -m pytest -q
