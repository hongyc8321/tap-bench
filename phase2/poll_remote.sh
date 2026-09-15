#!/usr/bin/env bash
cd ~/phase2_bundle/phase2
echo "===== run_all.log ====="
cat run_all.log 2>/dev/null
echo "===== latest gen log ====="
tail -3 gen_*.log 2>/dev/null
echo "===== processes ====="
pgrep -af "generate_probe" || echo "no generate_probe process"
pgrep -af "pip install" || true
echo "===== outputs so far ====="
ls gen_out/*/t2v 2>/dev/null | wc -l
du -sh ~/hf_cache 2>/dev/null
