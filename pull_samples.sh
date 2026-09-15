#!/usr/bin/env bash
set -e
S=/home/ubuntu
D=/home/rhap/nkst/samples
mkdir -p $D/{gt,cogvideox2b_orig,A0_finetuned,ltxvideo}

# GT references from the local probe set (what a real 24fps camera sees)
for cell in k08_w2.0_s01 k16_w4.0_s01; do
  for v in point shutter180; do
    src=/home/rhap/nkst/tap_bench/probe_set/wheel/${cell}_${v}
    n=$(ls $src/frame_*.png | wc -l)
    # encode GT png sequence to mp4 for easy viewing
    /home/rhap/nkst/.venv/bin/python - "$src" "$D/gt/${cell}_${v}.mp4" "$n" <<'EOF'
import sys, imageio.v2 as imageio, glob, os
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
frames = sorted(glob.glob(os.path.join(src, "frame_*.png")))[:n]
with imageio.get_writer(dst, fps=24, codec="libx264", quality=7, macro_block_size=1) as w:
    for f in frames:
        w.append_data(imageio.imread(f))
EOF
  done
done

pull() { scp -q -i /home/rhap/.ssh/id_ed25519 "ubuntu@36.103.236.54:$1" "$2"; }

# cogvideox2b original (pre-finetune)
for cell in k02_w0.5_s01 k08_w2.0_s01 k16_w4.0_s01; do
  pull $S/phase2_bundle/phase2/gen_out/cogvideox2b/t2v/${cell}_rep0.mp4 \
       $D/cogvideox2b_orig/${cell}.mp4
done

# A0 finetuned (same cells, direct A/B)
for cell in k02_w0.5_s01 k08_w2.0_s01 k16_w4.0_s01; do
  pull $S/phase3/gen_A0/cogvideox2b/t2v/${cell}_rep0.mp4 \
       $D/A0_finetuned/${cell}.mp4
done

# ltxvideo t2v vs i2v (motion freeze demo)
for cell in k08_w2.0_s01 k16_w4.0_s01; do
  pull $S/phase2_bundle/phase2/gen_out/ltxvideo/t2v/${cell}_rep0.mp4 \
       $D/ltxvideo/${cell}_t2v.mp4
  pull $S/phase2_bundle/phase2/gen_out/ltxvideo/i2v/${cell}_rep0.mp4 \
       $D/ltxvideo/${cell}_i2v.mp4
done

echo "== done =="
du -sh $D
find $D -name "*.mp4" | sort
