# Register Poetry virtualenv as a Jupyter kernel
register-kernel:
	poetry run python -m ipykernel install --user --name tcc-markov-chain --display-name "Python (tcc-markov-chain)"