# CRCPV Phase-0
Run in order (from this folder):
    pip install -r requirements.txt
    python scripts/run_demo.py            # bank + PASS log + replay (freezes test hash)
    python scripts/run_demo.py --leaky    # bank + FAIL log
    python scripts/run_audit.py
    python scripts/run_comparison.py
    python scripts/consolidate.py         # results/metrics/final_results.csv
