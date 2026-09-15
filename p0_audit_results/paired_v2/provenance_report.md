# 配置 Provenance 补充

## 已确认

- Phase 3 `eval_arm.sh` 对 A0-A5 使用同一入口：`generate_probe.py --model cogvideox2b --subset curve --gt ... --out gen_<ARM> --lora lora_<ARM>`，随后统一调用 `eval_gen.py`。
- 生成脚本中的 CogVideoX 固定规格为：模型仓库 `THUDM/CogVideoX-2b`、模型规格 FPS 8、720x480、49 帧、50 steps、guidance 6.0、float16；seed 为 `sha256(cell_key)` 前 4 字节，A0-A5 使用相同 seed 规则。
- A4 sharp 额外追加 `exposed with a 0-degree (instantaneous) shutter`；A4 blur 删除 `no motion blur` 并追加 `exposed with a 180-degree shutter causing motion blur`。因此 A4 与其他训练臂不是完全相同 prompt 协议。
- 日志显示各训练臂使用了模型缓存 revision `1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01`，但命令仍以 repo 默认路径加载，版本依赖和完整运行环境未写入 manifest。
- 旧生成脚本的断点续跑只检查 MP4 存在且非空，不检查 prompt、LoRA、代码或参数指纹。

## 未确认

- 每个 LoRA 目录实际加载的是最终权重还是 checkpoint-1500/2000；日志只显示 `loaded LoRA: lora_<ARM>`。
- A0-A5 的训练步数、训练数据 hash、学习率和随机种子是否完全一致；相关日志和 prompt 文件已备份，但尚未解析成结构化比较表。
- 训练视频数量、训练/验证划分和 held-out 速度区间是否完全一致。
- 旧 summary 的测量是否使用云端代码的同一 commit；当前源码无 Git commit provenance。

## 影响

配对 cell 一致、生成 seed 规则一致是强项，但 A4 的 prompt 介入、LoRA 选择规则和训练配置未完全追溯，必须在最终报告中作为实验限制。当前配对重测应定位为统一测量器下的描述性重分析，而不是已完成因果实验验证。
