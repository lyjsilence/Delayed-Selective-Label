# Delayed Selective Label

Reproduction of Scheduled Information Value for Delayed Selective Labels.

## Run


```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --workers 8
```

The full command runs 1,000 seeds, with 4,000 sequential decisions per seed. Outputs are generated locally in `results/`:

## Files

```text
config.json     Fixed DGP, policy parameters, and seeds
environment.py Independent random streams and delay calibration
policy.py       SF-SIV information calculations and decision loop
run.py          Run the seeds and write the table row
tests/          Feedback-timing and information-value checks
```
