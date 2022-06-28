@echo off
set PIPENV_VERBOSITY=-1
pipenv run python main.py --datapath data/crypto_processed/ --checkpointpath data/checkpoints/ --gencount 1500

exit