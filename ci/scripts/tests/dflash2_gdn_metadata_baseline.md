# DFlash2 GDN metadata 阶段 0 基线

本基线固定请求和采样参数，以独立进程覆盖以下四种配置，避免模型实例、图捕获及缓存状态相互污染：

| case | async scheduling | DFlash2 drafter FULL graph |
| --- | --- | --- |
| `async-off_full-graph-off` | 关闭 | 关闭 |
| `async-off_full-graph-on` | 关闭 | 开启 |
| `async-on_full-graph-off` | 开启 | 关闭 |
| `async-on_full-graph-on` | 开启 | 开启 |

默认固定参数为：batch 2、speculative tokens 8、生成 32 tokens、warmup 2 轮、测量 5 轮、seed 2026、FP16、max model length 256。采样使用 temperature 0，脚本会自动比较四种配置的每轮 token IDs。

## 运行

在安装当前 vLLM-Kunlun 与 `kunlun_ops` 的设备环境中，从仓库根目录运行：

```bash
python ci/scripts/tests/profile_dflash2_gdn_metadata.py \
  --target-model /path/to/target-model \
  --draft-model /path/to/dflash2-model \
  --output-dir /tmp/dflash2-gdn-baseline
```

脚本默认强制 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，使 TP=1 时 Python hook 与实际 `GPUModelRunner` 位于同一进程。这样可准确记录：

- `GPUModelRunner._build_attention_metadata` 调用次数和主机 wall time；
- `GDNAttentionMetadataBuilder.build()` 完整 build 次数和主机 wall time；
- 每轮端到端 latency、输出 token throughput；
- 四种配置下的确定性 token IDs。

模型无法在单卡运行时可设置 `--tensor-parallel-size N`，但多进程 executor 中 Python hook 可能只覆盖部分 rank；这时 build 次数和 CPU 时间不能作为正式结论，应以 TP=1 小模型复现或在 worker 内增加相同 hook。XProfiler 的 device timeline 仍可用于部署形态分析。

## XProfiler 采集

先启动 daemon：

```bash
/path/to/xprofiler -r 500 --xpu=0 -e /tmp/dflash2-gdn-baseline/trace_output -d
```

再设置 daemon 生成的 socket，并运行单个 case。一次只采一个 case，trace 更容易对齐：

```bash
export XPU_ENABLE_PROFILER_TRACING=1
export XPU_TRACING_OUTPUT_NAME=/path/to/xprofiler.sock
python ci/scripts/tests/profile_dflash2_gdn_metadata.py \
  --target-model /path/to/target-model \
  --draft-model /path/to/dflash2-model \
  --async-scheduling on \
  --full-graph on \
  --xprofiler \
  --output-dir /tmp/dflash2-gdn-baseline/async-on-full-on
```

`--xprofiler` 只包围 warmup 之后的测量区间。若要分别采集四象限，使用不同的 `trace_output` 目录重复执行单个 case。

## 结果与人工记录

每个 case 生成一个 JSON，`matrix-summary.json` 汇总核心指标及 token 一致性。XProfiler 中的以下指标需要人工读取并补入实验记录：

| 指标 | 读取方式 | baseline |
| --- | --- | --- |
| `_build_attention_metadata` CPU total / p95 | case JSON | 待采集 |
| GDN full build 次数 / metadata call | case JSON | 待采集 |
| D2H 次数与总时长 | XProfiler timeline | 待采集 |
| H2D 次数与总时长 | XProfiler timeline | 待采集 |
| metadata 后最大 device 空泡 | XProfiler timeline | 待采集 |
| decode 稳态平均 step latency | XProfiler timeline | 待采集 |
| output token throughput | case JSON | 待采集 |
| 四象限 token 一致性 | `matrix-summary.json` | 待采集 |

正式对比阶段 1/2/3 时必须复用同一模型、prompt、batch、参数和 profiling 窗口。不要把初始化、模型加载、首次图捕获混入稳态数据。

## 阶段 2：复用开启前的等价性验证

阶段 2 保持 `supports_update_block_table=False`。增加
`--verify-gdn-update` 后，每个 GDN group 仍执行完整 `build()`，脚本同时用首个
group 的 metadata 和后续 group 的 block table 调用 `update_block_table()`，逐项比较
device/CPU state indices，并检查公共 graph buffer 地址相同、group-specific state
buffer 地址不同：

```bash
python ci/scripts/tests/profile_dflash2_gdn_metadata.py \
  --target-model /path/to/target-model \
  --draft-model /path/to/dflash2-model \
  --num-speculative-tokens 3 \
  --verify-gdn-update \
  --output-dir /tmp/dflash2-gdn-stage2
```

验证模式会产生额外 clone 和 device 同步，只用于正确性检查，不能用其 latency 或
throughput 评价性能。每个 case 的 JSON 中
`metrics.gdn_update_equivalence.group_updates_checked` 必须大于 0；任一字段或地址不一致
会立即抛出异常。纯 speculative decode 的两个 CPU spec-state 字段没有消费者，优化后
`update_block_table()` 必须保持它们为 `None`；验证器对此检查“未物化”，其余 device
state 和 non-spec CPU state 仍与完整 build 逐项比较。

## 阶段 3：启用跨 KV group 复用

阶段 3 默认只为精确类型 `GDNAttentionMetadataBuilder`、`mamba_cache_mode` 为
`none/all` 且未启用 u-batching 的配置打开 vLLM 0.25.1 通用 metadata cache。可用
`--expect-gdn-reuse` 验证稳态每次 `_build_attention_metadata` 恰好只有一次完整 GDN
build，并且至少执行一次 block-table update：

```bash
python ci/scripts/tests/profile_dflash2_gdn_metadata.py \
  --target-model /path/to/target-model \
  --draft-model /path/to/dflash2-model \
  --num-speculative-tokens 3 \
  --expect-gdn-reuse \
  --output-dir /tmp/dflash2-gdn-stage3
```

对于当前每轮 10 个等价 GDN group 的模型，预期：

- `gdn_full_builds_per_metadata_call == 1`；
- `gdn_updates_per_metadata_call == 9`；
- `gdn_block_table_update.calls == group_updates_checked` 的阶段 2 对照值；
- 四象限 token IDs 仍完全一致。

阶段 2 的 `--verify-gdn-update` 会自动关闭真实复用，以保留后续 group 的完整 build
作为参考。若现场需要快速回退阶段 3，可在启动进程前设置：

```bash
export VLLM_KUNLUN_DISABLE_GDN_METADATA_REUSE=1
```

该开关只关闭跨 group metadata reuse，阶段 1 已完成的 CPU mirror 和异步传输优化仍然
保留。

## 扩展精度矩阵

`--precision-suite` 使用独立进程覆盖普通生成与 speculative generation 的更多调度和
图执行形态：

| case | 重点覆盖 |
| --- | --- |
| `regular-b1` | batch 1 普通 prefill/decode |
| `regular-b3-graph-padding` | 普通 decode 的 graph padding row |
| `spec-b1` | batch 1 prefill 与纯 speculative decode |
| `spec-b3-graph-padding` | speculative decode 的 graph padding row |
| `spec-b4-async` | 多 batch + async scheduling + FULL graph |
| `spec-b2-target-eager` | target eager + drafter 非 FULL graph |
| `spec-b2-mixed-prefill` | chunked prefill 与 speculative decode 混合 batch |

本轮矩阵不改变或比较 `mamba_cache_mode`。每个 case 除了保存 token IDs，还会记录
`metrics.gdn_metadata_paths`；如果配置没有实际命中声明的 prefill、decode、pure-spec、
mixed 或 graph-padding 路径，带 `--expect-gdn-reuse` 的 candidate worker 会直接失败，
避免名义覆盖。历史 reference 只负责提供已知正确 token；由于旧版 metadata 的 padding
字段形态可能不同，不对它应用当前实现的路径命中断言。

四象限内部一致只能证明不同执行模式彼此一致，不能排除它们共同出错。扩展矩阵应先用
已知正确的 `c4c2b50` 代码和同一份最新版脚本生成外部基线，再在当前分支逐 token
比较。建议把脚本复制到代码 checkout 之外，确保两个 revision 执行完全相同的 case：

```bash
cp ci/scripts/tests/profile_dflash2_gdn_metadata.py /tmp/profile_dflash2_gdn_metadata.py

# 在安装了已知正确 c4c2b50 的环境中生成 reference；不要加 --expect-gdn-reuse。
python /tmp/profile_dflash2_gdn_metadata.py \
  --target-model /path/to/target-model \
  --draft-model /path/to/dflash2-model \
  --num-speculative-tokens 3 \
  --warmup-iterations 1 \
  --iterations 2 \
  --precision-suite \
  --output-dir /tmp/gdn-precision-reference

# 切回并安装本分支后，与 reference 逐 case、逐轮、逐请求、逐 token 比较。
python ci/scripts/tests/profile_dflash2_gdn_metadata.py \
  --target-model /path/to/target-model \
  --draft-model /path/to/dflash2-model \
  --num-speculative-tokens 3 \
  --warmup-iterations 1 \
  --iterations 2 \
  --precision-suite \
  --expect-gdn-reuse \
  --reference-dir /tmp/gdn-precision-reference \
  --output-dir /tmp/gdn-precision-candidate
```

`precision-summary.json` 中以下字段必须成立：

- `all_cases_deterministic == true`；
- `all_cases_match_external_reference == true`；
- 每个 case 的 `gdn_metadata_paths.total` 包含其要求的路径；
- 当前 10 个等价 GDN group 的模型中，每个 case 应保持一次 full build 加九次 update，
  即 `gdn_full_builds_per_metadata_call == 1`、
  `gdn_updates_per_metadata_call == 9`。

若只需为原四象限增加外部基线校验，也可直接给原命令增加
`--reference-dir /path/to/stage0-json-dir`；文件名必须与对应 case 相同。
