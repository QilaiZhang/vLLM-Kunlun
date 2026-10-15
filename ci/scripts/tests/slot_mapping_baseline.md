# Qwen3.5 + DFlash2 slot mapping 基线采集

本轮只采集基线，不改动 slot mapping 计算。请在已能正常运行 Qwen3.5 + DFlash2 的昆仑环境中执行，保留同一 target/draft、精度、设备可见性和运行参数供后续优化对比。

## 脚本与准备

主入口为 `ci/scripts/tests/profile_dflash2_gdn_metadata.py`，新增选项 `--profile-slot-mapping`。它依赖同目录的 `profile_slot_mappings.py`，复制到远端时必须一起复制；不要只复制主脚本。请保留当前 `1ff9154` 的 GDN metadata 实现作为基线，不混入后续 slot mapping 优化。

后续正确性验收统一使用此 profile 入口，以 `--reference-dir` 比较优化前后的逐轮
token IDs，同时检查实际 group 调用计数、GDN metadata 复用和路径覆盖。本轮新增的
`test_` 脚本已按要求移除，后续不再新增；未覆盖的配置仍需通过 profile 单独采集。

在仓库根目录、使用现有推理环境的 Python 执行。以下示例设置可读的模型路径变量：

```bash
export TARGET_MODEL=/path/to/Qwen3.5-target
export DRAFT_MODEL=/path/to/DFlash2-draft
export BASELINE_DIR=/tmp/qwen35-slot-baseline
export VLLM_USE_V2_MODEL_RUNNER=0
```

主脚本默认使用 TP=1，并设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，让 Python hook 与 V1 runner 在同一进程。若模型无法单卡运行，可调整 `--tensor-parallel-size`，但当前 Python 计数可能不覆盖 worker rank；这种数据不能当作完整 slot mapping 基线。脚本不会改变现有设备可见性，请沿用已验证的 XPU 环境设置。

不要添加 `--verify-gdn-update`：等价性验证会产生额外 clone 和同步，脚本会拒绝将它混入 slot mapping 基线。

## 1. DFlash2 四象限

运行 async scheduling on/off 与 drafter FULL graph on/off 的四种组合，每种在独立进程中 warmup 后测量：

```bash
python ci/scripts/tests/profile_dflash2_gdn_metadata.py \
  --target-model "$TARGET_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --profile-slot-mapping \
  --num-speculative-tokens 3 \
  --batch-size 2 --max-num-seqs 4 \
  --max-model-len 256 --max-tokens 32 \
  --warmup-iterations 2 --iterations 5 \
  --require-gdn-path prefill --require-gdn-path pure_spec \
  --output-dir "$BASELINE_DIR/spec"
```

`num-speculative-tokens=3` 与现有精度矩阵一致。若实际部署使用其他 K，应额外采集其对应配置，并在之后保持一致。默认 dtype 为 float16；实际使用其他精度时显式设置 `--dtype`。若当前模型已确认有多个等价 GDN group，可加 `--expect-gdn-reuse` 检查每个 metadata call 只有一次完整 build 且确实执行 update。

生成 `spec/*.json` 和 `spec/matrix-summary.json`。路径断言确保测量区间实际命中 prefill 与 pure-spec，而不是只完成了模型初始化。

## 2. 普通 prefill/decode

```bash
python ci/scripts/tests/profile_dflash2_gdn_metadata.py \
  --target-model "$TARGET_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --profile-slot-mapping --disable-speculative \
  --async-scheduling off --full-graph off \
  --batch-size 2 --max-num-seqs 4 \
  --max-model-len 256 --max-tokens 32 \
  --warmup-iterations 2 --iterations 5 \
  --require-gdn-path prefill --require-gdn-path decode \
  --output-dir "$BASELINE_DIR/regular"
```

`--full-graph` 控制 drafter；target 图模式由正常配置决定，不表示关闭 target graph。无 speculative 模型时该 drafter 开关不会参与推理。

## 3. Chunked prefill 与 speculative decode 混合

```bash
python ci/scripts/tests/profile_dflash2_gdn_metadata.py \
  --target-model "$TARGET_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --profile-slot-mapping \
  --async-scheduling off --full-graph on \
  --num-speculative-tokens 3 \
  --batch-size 2 --max-num-seqs 4 \
  --max-model-len 512 --max-num-batched-tokens 64 \
  --enable-chunked-prefill --prompt-set staggered \
  --max-tokens 24 --warmup-iterations 2 --iterations 5 \
  --require-gdn-path prefill --require-gdn-path mixed_prefill_spec \
  --output-dir "$BASELINE_DIR/mixed"
```

若路径断言失败，保留日志并调整请求/调度参数，直到实际命中；不要去掉断言后把它标成 mixed 基线。

## XProfiler trace

CPU JSON 不会给出 kernel 时间或 H2D/D2H 的真实耗时。需要 trace 时沿用设备环境中现有的 xprofiler daemon/socket，例如：

```bash
/path/to/xprofiler -r 500 --xpu=0 \
  -e "$BASELINE_DIR/trace-spec-on" -d
export XPU_ENABLE_PROFILER_TRACING=1
export XPU_TRACING_OUTPUT_NAME=/path/to/daemon-generated.sock
```

用上述 DFlash2 命令再采集一个独立 case，增加 `--async-scheduling on --full-graph on --xprofiler`，输出到新的目录。一次 trace 只采一个 case；普通生成和 mixed 场景也分别采集。daemon 的 `--xpu` 必须与实际可见设备一致，socket 必须使用 daemon 实际生成的路径。

`--xprofiler` 只包围 warmup 之后的测量区间。开启 trace 的结果与未开启 trace 的结果分开保存，避免将 trace 本身的开销误认为计算变化。读取：

- `compute_slot_mapping` 原生入口中的临时 metadata H2D 和可能的等待。
- slot mapping kernel 设备时间及其后的设备空泡。
- GDN prefill/mixed 中的 `has_initial_state` D2H。
- 每个稳态 step 的总耗时；不计模型加载、graph capture 或 warmup。

## JSON 指标解释

每个 case 的 `environment` 记录加载的 vLLM/plugin/kunlun_ops/torch 版本、模块路径、可发现的源码 git revision/dirty 状态、native extension 路径、可见设备和两个采集脚本的 SHA256。wheel 无源码 checkout 时 revision 可能为 null；模块源码 revision 也不能替代编译库的版本确认。

`runtime` 记录实际 V1 runner 类、mamba cache mode、每组 spec 类型、层数、manager/kernel block size、block table shape/stride、slot capacity 和 CP 配置。没有 `SlotMappingMode` 的版本会记录 null，不通过模型名猜测是否应跳过。

`metrics.slot_mapping` 包含：

| 字段 | 意义 |
| --- | --- |
| `batch` | `MultiGroupBlockTable.compute_slot_mapping` 的次数和 CPU total/mean/p50/p95/max |
| `group_method` | 所有 `BlockTable.compute_slot_mapping` 方法的计时，包含之后可能跳过的 group |
| `native_op_entry` | `kunlun_ops.compute_slot_mappings` Python 入口至返回的主机时间，包含 custom-op dispatch、C++ host 工作和其中发生的等待 |
| `per_group` | 按实际 cache spec 分组的 method/native 计时和调用次数 |
| `unattributed_native_op_entry` | 不位于已登记 V1 group 方法中的 native 调用，单独保留，不能强行归给 target GDN/Attention |
| `batch_shapes` | 从张量 shape 和 host request 数得到的 batch 分布，不读取设备 tensor 内容 |

这些时间是嵌套的，不能相加。工具不增加 `.cpu()`/`.item()`/device synchronize，但 Python 计时与记录仍有开销；后续必须使用同一份脚本比较各 revision，最终端到端结论另用关闭 instrumentation 的运行验证。

跳过 GDN 优化后，它的 `method.calls` 可以仍大于 0，但 `native_op_entry.calls` 应为 0。当前基线不预设它已经跳过，也不预设每步固定有 10 个 GDN group。

当前 native hook 对应 `kunlun_ops.compute_slot_mappings`。后续若引入新的单 group/多 group 接口，必须同时更新对应入口 hook，再使用 native 调用次数判断是否跳过，避免接口换名被误判为没有计算。

## 回传结果与验收

请回传 `spec/`、`regular/`、`mixed/` 的 JSON 和运行日志；有 trace 时同时提供 trace 文件或 H2D/D2H、等待与设备空泡的摘要。默认命令中每个 case JSON 会保留逐轮 token IDs，矩阵汇总检查确定性；这些是性能基线输出，不替代之后的外部 reference 精度验证。

在同一硬件和参数上至少重复一组采集，记录波动，再据此确定任务 5～8 的收益与优先级。

## 2026-10-14 用户回传的主机基线

读取 `D:/Downloads/qwen35-slot-baseline` 中的六个 case JSON 和三份矩阵汇总。实际 target/draft 为 `Qwen3.8-27B` / `Qwen3.8-27B-DFlash2`，target `model_type=qwen3_5_text`，包含 48 个 linear-attention 层和 16 个 full-attention 层。TP=1、float16、V1 `GPUModelRunner`、`mamba_cache_mode=none`。以下为一次五轮测量结果，未读取 device trace。

加载的包版本：vLLM `0.25.1`、vllm-kunlun `0.25.1.dev0`、kunlun_ops `0.1.7+adaa0b25`、torch `2.9.0+cu128`。均从 site-packages 加载，源码 git revision 未提供，不能仅靠版本号证明部署代码等于本地 `1ff9154`。

| 场景（async / drafter FULL） | batch calls | native calls | GDN native calls | batch mean / p95（ms） | 输出 tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| spec off / off | 60 | 900 | 600 | 1.390 / 1.416 | 62.044 |
| spec off / on | 60 | 900 | 600 | 1.336 / 1.358 | 71.846 |
| spec on / off | 65 | 975 | 650 | 1.373 / 1.407 | 57.064 |
| spec on / on | 65 | 975 | 650 | 1.360 / 1.414 | 65.546 |
| regular off / off | 160 | 640 | 480 | 0.392 / 0.429 | 39.780 |
| mixed off / on | 85 | 1275 | 850 | 1.359 / 1.405 | 12.483 |

确认的事实：

- Spec/mixed 有 10 个 GDN group、5 个 FullAttention group，每个 batch 仍对全部 15 个 group 调用原生算子。GDN 占原生调用数的 2/3；regular 有 3 个 GDN group、1 个 Attention group。
- Spec/mixed 的 GDN metadata 已达到每次一次 full build、九次 update；regular 为一次 full build、两次 update。metadata 复用没有省掉 GDN slot mapping 调用。
- Spec 的 Attention manager block size 为 800、kernel block size 为 16（50 个 kernel block/manager block）；regular 为 784/16（49 个）。混合路径 GDN block size 为 512，spec 为 256。后续不得硬编码 block size 或 group 数。
- 所有 case 的 `slot_mapping_mode` 为 null：实际 0.25.1 没有本地 main 的 mode 字段，跳过 GDN 必须依据实际 KV cache spec 建立 group 标记。
- Spec 四象限均命中 prefill/pure-spec/graph padding；regular 命中 prefill/decode；mixed 命中 mixed prefill/spec、variable-spec 和 graph padding。
- 三份汇总的 `all_tokens_match=true`，但未提供外部 reference，对照不同 revision 的精度验证仍留到后续阶段。

计时边界与可重复性：

- 这些是带 instrumentation 的主机 wall time，不能由此确定同步发生在哪个 SDK 调用，或宣称获得对应端到端加速；trace 由用户自行分析。
- 回传脚本 SHA256：主脚本 `fb4d6eca3d466eb317682827cf0ec933a102d3e801f15d70fdf1eab7d0a03a47`，helper `0d706a94071067bb5b3c8e7366c1e45ce95ef008b17f76c113587f7851a33f8f`。用户确认现场文件仅缺少末尾换行，不影响执行语义。后续采集复用现场相同脚本，保留实际 SHA256；本地提交时的格式化也会改变文件字节。
- 当前只收到 JSON，无运行日志和原始 trace。用户分析 trace 后已确认存在同步操作，后续出现 device 空泡；尚未提供具体同步 API 和各项时长，不能据此将全部等待归给 `do_host2device`。

下一优先项仍是按实际 spec 跳过 GDN slot mapping，再缓存 Attention 的 `block_sizes`。性能收益需通过同参数、同采集工具的优化后数据验证。

## 第一阶段：仅跳过 GDN/Mamba slot mapping

本阶段修改 `vllm_kunlun/v1/worker/block_table.py` 和
`vllm_kunlun/registration/compat_patches.py`。现场插件与当前分支的完整 hook 注册表
不同，部署时只应用这两个文件的 stage1 差异。采集脚本沿用现场基线版本；本阶段尚未
缓存 Attention 的 `block_sizes`，也未修改原生算子。

### 现场按差异部署

用户已确认：回退版 `compat_patches.py` 没有 `_indexer_applied`、
`_apply_indexer_patch` 或 `deepseek_v2` 注册项。此前直接替换整份文件，额外启用了
当前分支的 MLA indexer hook；全局 dispatcher 在模块加载后尝试导入远端 indexer，
才暴露 `kv_spans_from_batches` 缺失。这是整文件部署引入的额外兼容改动。

先恢复现场 stage1 之前能运行的两个文件，再对照本分支相对 `fb88875` 的差异，
只添加 stage1 改动：

- `block_table.py`：新增 `functools` import、`_set_slot_mapping_enabled`、
  `patch_slot_mapping_runner`，并在原 `_compute_slot_mapping` 入口添加跳过判断。
- `compat_patches.py`：只更新 `_gpu_model_runner_applied` 和
  `_apply_gpu_model_runner_patch` 两个函数，保留现场其余函数和 `DEFAULT_HOOKS`。

安装后重启推理进程。不要整份替换现场 `compat_patches.py`；本阶段无需新增 MLA
hook 或替换 indexer。为规避整份替换而新增的安装脚本和 import 调整已按用户要求撤销。

0.25.1 没有 `slot_mapping_mode`，因此在 V1 input batch 创建/重建后根据实际
`MambaSpec` 标记 group。已有 `SlotMappingMode.NONE` 的版本也保留其跳过语义。
Unknown/mixed spec 保持原计算，buffer 重建或 group 类型变化时刷新标记。

### 设备端对比

复用前三组基线命令及模型、参数，仅调整输出目录并添加外部 reference：

| 场景 | `--output-dir` | `--reference-dir` |
| --- | --- | --- |
| DFlash2 四象限 | `$BASELINE_DIR/stage1-spec` | `$BASELINE_DIR/spec` |
| regular | `$BASELINE_DIR/stage1-regular` | `$BASELINE_DIR/regular` |
| mixed | `$BASELINE_DIR/stage1-mixed` | `$BASELINE_DIR/mixed` |

本次回传数据对应 target `/nas/disk1/Qwen3.8-27B` 和 draft
`/nas/disk1/Qwen3.8-27B-DFlash2`，后续比较保持这两个模型。可增加
`--expect-gdn-reuse`，确认 slot mapping 改动没有破坏已有 GDN metadata 复用。

验收：

- `per_group` 中所有 `gdn_mamba.native_op_entry.calls` 为 0；其 method calls
  可以仍大于 0，因为 MultiGroup 循环仍调用每个 group 方法。
- Spec/mixed 每个 batch 的 native 调用从 15 次变为 5 次，regular 从 4 次变为
  1 次；每个有效 Attention group 的调用次数应与 batch calls 一致。
- 所有 case 的外部 reference token IDs 一致，原有路径覆盖仍命中。
- 对比 batch CPU mean/p95、Attention native 时间和端到端指标；trace 由用户分析，
  确认同步次数及空泡的变化。不要用减少的调用比例直接推算端到端收益。

## 2026-10-14 stage1 结果

已读取 `D:/Downloads/qwen35-slot-baseline/stage1-spec`、`stage1-regular`、
`stage1-mixed` 中的六个 case JSON 和三份汇总，并分别对照原 `spec`、`regular`、
`mixed` 的同名 case。逐项比较 config、prompts、runtime、environment、batch_shapes
和 batch calls，均一致；采集脚本 SHA256、包版本及执行环境也一致。

六个 case 的全部五轮 token IDs 均与原基线完全一致，三份汇总的
`all_tokens_match_external_reference=true`。路径计数未变；GDN metadata 仍为
spec/mixed 每次 1 个 full build + 9 个 update，regular 为 1 + 2。

所有 GDN group 的 native calls 均为 0，Attention 各组 native calls 均等于 batch
calls，未归因 native calls 为 0。Spec off/off、off/on 的总 native calls 为
900→300，on/off、on/on 为 975→325；mixed 为 1275→425，regular 为 640→160。
每步调用分别从 15→5、4→1，符合 stage1 预期。

以下均为带 instrumentation 的主机 wall time；吞吐变化为本次两组运行的观察值。

| 场景（async / drafter FULL） | batch mean 基线→stage1（ms） | batch p95 基线→stage1（ms） | mean 降幅 | 吞吐变化 |
| --- | ---: | ---: | ---: | ---: |
| spec off / off | 1.390→0.563 | 1.416→0.586 | 59.5% | +1.39% |
| spec off / on | 1.336→0.564 | 1.358→0.592 | 57.8% | +0.34% |
| spec on / off | 1.373→0.572 | 1.407→0.608 | 58.4% | +0.99% |
| spec on / on | 1.360→0.566 | 1.414→0.587 | 58.4% | +1.25% |
| regular off / off | 0.392→0.149 | 0.429→0.167 | 62.0% | +0.91% |
| mixed off / on | 1.359→0.567 | 1.405→0.599 | 58.3% | +0.49% |

Spec 每步减少约 0.77～0.83 ms，mixed 减少约 0.79 ms，regular 减少约 0.24 ms。
按测量窗口换算，每轮 slot mapping 主机时间减少约 7.8～13.5 ms；它只占完整生成
耗时的一小部分，不能把 58%～62% 的局部降幅当成端到端加速。
每个 case 只有一次运行中的五轮样本，尚无独立重复测量或关闭 instrumentation
后的结果；尤其 FULL graph 和 mixed 的吞吐变化仍可能落在运行波动范围内。

剩余开销支持继续任务 6：spec/mixed 每个 batch 的 Attention method 合计约
0.451～0.460 ms，native 入口合计约 0.135～0.142 ms，两者相差约 0.313～0.319 ms；
regular 分别为 0.111、0.037 ms，相差约 0.074 ms。该差额包含 `torch.tensor`
创建及其他 Python/instrumentation 开销，不能全部归因于小张量上传。
同一 Attention 组的 method/native 主机时间比基线略高，因此只确认总调用和总
主机时间下降，没有证据表明保留的单个 Attention 调用本身已加速。

下一项为缓存有效 Attention group 的 `block_sizes`，继续单项对比。
原生 C++ metadata 分配/H2D 仍存在，须后续原生入口优化处理。
本次未提供 stage1 trace 摘要，不推断设备同步次数、同步时长或空泡已经消除。

## 第二阶段：缓存设备 block_sizes

Stage1 提交为 `6f3aa1d`。本阶段运行代码仅修改
`vllm_kunlun/v1/worker/block_table.py`，以该提交作为差异基准；现场已经部署 stage1
时，仅合入本阶段的 block table 差异，重启推理进程即可。

每个有效 group 缓存一个 int32 设备张量 `[table.block_size]`。在 input batch
创建/重建后的 group 配置阶段准备，稳态直接复用；直接调用 BlockTable 的路径支持
首次调用时初始化。缓存 key 为 kernel block size 与设备，二者变化时重建。
新 input batch 中的新 table 自带独立缓存；同设备 buffer 替换时可以继续复用该
标量，因为本阶段没有缓存输入/输出指针、stride 或 CP 参数。
GDN 与 `SlotMappingMode.NONE` group 不创建此张量。

本阶段保留 Attention 原生调用次数及现有参数；C++ 内部 pointer/stride metadata
分配和 H2D 仍未优化，因此仍可能看到同步。原始 profile 脚本保持不变。
以下指令对照已验收的 stage1，正确性通过外部 reference token 比较与路径断言验收：

```bash
export TARGET_MODEL=/nas/disk1/Qwen3.8-27B
export DRAFT_MODEL=/nas/disk1/Qwen3.8-27B-DFlash2
# 改为现场保存基线和 stage1 结果的目录。
export BASELINE_DIR=/tmp/qwen35-slot-baseline
export VLLM_USE_V2_MODEL_RUNNER=0

python ci/scripts/tests/profile_dflash2_gdn_metadata.py \
  --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
  --profile-slot-mapping --expect-gdn-reuse \
  --num-speculative-tokens 3 --batch-size 2 --max-num-seqs 4 \
  --max-model-len 256 --max-tokens 32 \
  --warmup-iterations 2 --iterations 5 \
  --require-gdn-path prefill --require-gdn-path pure_spec \
  --reference-dir "$BASELINE_DIR/stage1-spec" \
  --output-dir "$BASELINE_DIR/stage2-spec"

python ci/scripts/tests/profile_dflash2_gdn_metadata.py \
  --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
  --profile-slot-mapping --expect-gdn-reuse --disable-speculative \
  --async-scheduling off --full-graph off \
  --batch-size 2 --max-num-seqs 4 --max-model-len 256 --max-tokens 32 \
  --warmup-iterations 2 --iterations 5 \
  --require-gdn-path prefill --require-gdn-path decode \
  --reference-dir "$BASELINE_DIR/stage1-regular" \
  --output-dir "$BASELINE_DIR/stage2-regular"

python ci/scripts/tests/profile_dflash2_gdn_metadata.py \
  --target-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
  --profile-slot-mapping --expect-gdn-reuse \
  --async-scheduling off --full-graph on --num-speculative-tokens 3 \
  --batch-size 2 --max-num-seqs 4 --max-model-len 512 \
  --max-num-batched-tokens 64 --enable-chunked-prefill --prompt-set staggered \
  --max-tokens 24 --warmup-iterations 2 --iterations 5 \
  --require-gdn-path prefill --require-gdn-path mixed_prefill_spec \
  --reference-dir "$BASELINE_DIR/stage1-mixed" \
  --output-dir "$BASELINE_DIR/stage2-mixed"
```

验收要求：三个汇总的 `all_tokens_match_external_reference=true`；GDN native
calls=0，spec/mixed 每步有效 native calls=5，regular=1；GDN 复用仍为 1+9 / 1+2。
对比 batch/group method/native 主机时间和端到端指标，不能由代码改动预设收益。
相同布局的稳态 trace 中，slot mapping 调用前反复出现的 block_sizes 张量创建应
消失；native 入口内部的 metadata 上传仍可能存在。trace 继续由用户分析。

本地已完成语法、静态检查及缓存生命周期代码核对。设备结果见下节；用户重新回传
spec 输出后，六个场景的 profile 输出回归和主机指标核验已完成。

## 2026-10-15 stage2 结果（更新回传后）

已读取更新目录中的全部六个 case 和三份汇总，并直接比较每个 case 的五轮 token
IDs：六项均与对应 stage1 完全一致。三份汇总的
`all_tokens_match_external_reference=true`，各 case 内五轮输出亦一致。

全部六个 case 的 config、runtime、prompts、batch_shapes、batch calls 与路径计数
均与 stage1 一致；环境记录仅 Python executable 字符串多了一个 `/`，即
`/opt/vllm_kunlun//bin/python`，包版本、设备、脚本 SHA 和环境变量均相同。
GDN native calls 仍全为 0，Attention 每组 native calls=batch calls，未归因调用=0。
Regular 为 160 batch/160 native，mixed 为 85 batch/425 native。
Spec async off 两项均为 60 batch/300 native，async on 两项为 65 batch/325 native。
Spec/mixed 每步有效 native=5，regular=1；GDN metadata 复用仍分别为 1+9、1+2。

| 场景 | batch mean stage1→stage2（ms） | batch p95 stage1→stage2（ms） | mean 降幅 | 吞吐 stage1→stage2（tokens/s） |
| --- | ---: | ---: | ---: | ---: |
| spec off / off | 0.563→0.290 | 0.586→0.305 | 48.5% | 62.908→62.759 |
| spec off / on | 0.564→0.284 | 0.592→0.308 | 49.6% | 72.089→72.205 |
| spec on / off | 0.572→0.295 | 0.608→0.314 | 48.4% | 57.632→57.886 |
| spec on / on | 0.566→0.293 | 0.587→0.319 | 48.2% | 66.367→66.658 |
| regular off / off | 0.149→0.097 | 0.167→0.126 | 35.1% | 40.140→40.048 |
| mixed off / on | 0.567→0.290 | 0.599→0.314 | 48.8% | 12.544→12.477 |

Spec 每个 batch 减少约 0.273～0.279 ms；regular 减少约 0.052 ms，mixed 约 0.277 ms。
Attention method 与 native 合计时间之差分别从 0.074→0.013 ms、
0.315→0.052 ms，符合去除逐步小张量创建开销的方向，但不是设备同步归因证据。
Native 平均主机时间 regular 为 0.037→0.045 ms，mixed 为 0.028→0.026 ms。
端到端吞吐本次分别变化 -0.23%、-0.53%，尚无重复测量，不能宣称获得端到端收益，
也不能仅由这次差异认定缓存造成稳定退化。Trace 仍由用户分析。
Spec 四项吞吐分别变化 -0.24%、+0.16%、+0.44%、+0.44%，也未建立稳定的端到端
收益证据。Spec Attention method 每个 batch 合计约 0.180～0.189 ms，native 合计约
0.128～0.136 ms，二者差额约 0.052～0.053 ms，已明显缩小。

此前收到的 spec 文件为 schema1、缺少 slot 指标且 max_num_seqs=2；用户确认运行
命令已传对应参数后，重新回传远端输出。当前四项 case 均为 `schema_version=2`，
`profile_slot_mapping=true`、`max_num_seqs=4`，环境、runtime、路径、slot 指标及
外部 reference 汇总完整。此前不符合比较条件的 spec 数据已被本次结果替代，
无需继续补采该矩阵；现有证据不用于反推此前文件不一致的具体原因。

任务6已通过本次 TP1/FP16/cache mode none 六个场景的 profile 输出回归与主机
指标验收。2026-10-15 用户检查当前 profile 后确认此前关注的同步已消失；原始 trace
由用户保管，本记录采用用户观察，不外推为所有配置或整个推理过程都不存在同步。
该观察表明当前采集路径的同步优化目标已达到，C++ 源码仍有 metadata 上传，
但不能仅凭该调用存在就认定它仍构成同步瓶颈。

建议本轮实现收敛在 stage2，任务7/8保留为条件性候选，暂不修改原生接口或合并
group。当前 spec/mixed batch 主机 mean 约 0.284～0.295 ms，regular 约 0.097 ms；
按测量窗口累加后折算，每轮 slot mapping 主机时间约占整轮生成 0.14%～0.39%，
且含 instrumentation，不能直接当成可兑现的端到端收益。
后续优先在实际部署配置下重复测量关闭 instrumentation 的端到端性能，并按需要
扩大 profile 正确性覆盖；仅当 trace 再次显示 slot mapping 位于关键路径，或更多
有效 Attention group/并发配置显著放大其开销时，再实施原生 metadata 或多组优化。
