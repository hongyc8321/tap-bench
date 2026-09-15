# Phase 2 部署指南（AutoDL 单卡 4090）

## 0. 上传包
本地 `phase2_bundle.tar.gz`（约 50–80MB）内含：
- `phase2/` 脚本 + requirements
- `gt_curves.csv`（提示词与 GT 参数）
- `first_frames/`（I2V 条件首帧，本阶段暂不使用但一并携带）

## 1. 租用与初始化（约 5 分钟）
1. AutoDL 租 **RTX 4090 / 24GB**，镜像选 PyTorch ≥2.3 + CUDA 12.1。
2. 上传 `phase2_bundle.tar.gz` 到 `/root/autodl-tmp`，解压：
   ```bash
   cd /root/autodl-tmp && tar xzf phase2_bundle.tar.gz && cd phase2
   ```
3. 内存盘放 HF 缓存（AutoDL 数据盘较小可跳过）：
   ```bash
   export HF_HOME=/root/autodl-tmp/hf_cache
   ```

## 2. 一键运行
```bash
nohup bash run_all.sh > run_all.log 2>&1 &
tail -f run_all.log
```
- 自动：建 venv → 装依赖（走 hf-mirror）→ 按序生成三个模型 → 统一评测。
- 断点续跑：脚本与 generate_probe.py 均可重复执行，已生成的 mp4 跳过。

## 3. 只跑部分模型
```bash
bash run_all.sh cogvideox2b          # 最快出第一条曲线（~2-3h）
bash run_all.sh cogvideox2b ltxvideo # 前两条（追加 ~1.5h）
```

## 4. 取回结果
只需要小文件（视频可留云端）：
```bash
# 本地执行
scp -P <port> root@<host>:/root/autodl-tmp/phase2/eval_out/* ./phase2_results/
```
`eval_out/` 内容：
- `summary.txt` — 头条数字：各模型整体反转率 + 分箱表
- `nyquist_curves.png` — 实测时域奈奎斯特曲线（RR vs f_tex/fps）
- `nyquist_bins.csv` / `nyquist_rows.csv` — 明细

## 5. 预算参考（curve 子集，153 cells/模型）
| 模型 | 单条耗时 | 总时长 | 费用(¥1.98/h) |
|---|---|---|---|
| cogvideox2b | ~70s | ~3h | ~¥6 |
| ltxvideo | ~40s | ~1.7h | ~¥3.5 |
| wan13b | ~3min | ~7.5h | ~¥15 |

三模型全跑 ≈ ¥25。出错重试建议先 `cogvideox2b` 单模型验证全链路。

## 6. 常见问题
- **OOM**：加 `--offload` 重新运行（run_all.sh 默认不开，需手动改）。
- **diffusers 版本报错**：确认 ≥0.34（LTX/Wan 管线需要）。
- **下载慢**：确认 `HF_ENDPOINT=https://hf-mirror.com` 已生效。
- **某条 mp4 测量失败**：eval 会跳过并在 rows 里缺行，不影响整体。
