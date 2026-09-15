#!/usr/bin/env bash
cd /home/rhap/nkst/tap_bench
exec ../.venv/bin/python measure_angle.py --root probe_set --results gt_measurements.csv --selftest \
  > full_measure.log 2>&1
