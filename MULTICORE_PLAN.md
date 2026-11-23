# PanFlat v1 真多核并行改造计划（`MULTICORE_PLAN.md`）

> **目的**：解决 `panflat build --threads N`（如 `-t 64`）在真实染色体多单倍型数据（例如 33 单倍型梨属 Chr15A）上长期只跑单核（~100% CPU）的问题，在**严格保证串行与并行输出逐字节一致（规格 T22）**、**不影响任何现有功能**且**不引发 IPC / COW 内存爆炸**的前提下，让核心计算阶段真正吃满多核。
>
> **状态**：已完成白盒代码与 `mappy` C 绑定级深度审查，并将全部正确性、IPC 序列化、COW 脏页与 `fork` 线程安全修正项合入本方案（尚未改动核心业务代码与规范文档）。

---

## 1. 现状瓶颈诊断

逐一审计当前代码库与上游 `mappy` 源码后，`panflat build` 全流程各阶段的并行现状如下：

| 阶段 | 代码位置 | 当前状态 | 为什么跑单核 / 瓶颈分析 |
|---|---|---|---|
| **1. 输入加载与预检** | `panflat/gfa_io.py::load_and_validate_inputs` | 单核 | 顺序流式读盘、算 SHA-256、构建紧凑 NumPy 存储；主要受磁盘 I/O 与解析带宽限制 |
| **2.1–2.3 着丝粒定界与处理** | `panflat/centromere.py::process_centromere_regions` | 单核 | `bypass` 很快；`extract-islands` 时 Step B（`mappy` 50 bp 减法）和 Step C（15-mer 滑窗重复屏蔽与切片）单核串行 |
| **2.4 倒位归一化 / 加权 LIS / 偏离段提取** | `panflat/linearizer.py` (L2865–L3452) | **单核** | 对 32 条分支 contig 逐条循环执行归一化、投影、LIS、微变异过滤和局部重锚，**彼此完全独立但未并行** |
| **2.5a 通用 repeat-guard 滑窗筛查** | `panflat/linearizer.py` (L3508–L3570) | **单核** | 对全部 `terminal_groups` 逐个做候选提取、15-mer 滑窗屏蔽与环折叠，完全串行 |
| **2.5b 平行分支组递归与聚类** | `panflat/linearizer.py` (L3573–L3615) | **伪多核（GIL）** | 整个项目**唯一**使用 `config.threads` 的地方，但用的是 Python `ThreadPoolExecutor`；受 Python 3.12 **GIL** 限制，纯 Python 递归/并查集/去重/窗口校验同一时刻只能跑 1 核 |
| **3. 写出与 `--qc` 冗余分析** | `panflat/verify_qc.py::run_qc` (L171–L398) | **完全单核** | `--qc` 做 Stage 1（全部 ≥150 bp 插入片回比整条连续参考）+ Stage 2（`2 × ceil(log2(N))` 轮建 `asm20` 及 `k=13, w=5, mid_occ=1000` 短索引并互比），**未用 `ThreadBuffer` 并发 `map()`**，在高重复植物染色体上极易数小时吃满单核 |
| **4. 最终不变量校验** | `panflat/verify_qc.py::verify_build_outputs` | 单核 | 流式逐块核对 Invariant I/II/III 和 GFA SHA-256；顺序 I/O 为主 |

**结论**：真实数据跑 `--threads 64 --qc` 时，CPU 长期停在 100%（单核）主要由三大计算瓶颈造成：**Stage 3 (`--qc`)**、**Stage 2.5 (repeat-guard + 分支组求解)**、**Stage 2.4 (多单倍型倒位与 LIS)**。

---

## 2. 核心设计约束（红线）

### 2.1 T22 严格确定性与全局编号顺序契约（不可妥协）
- `--threads 1` 与 `--threads N` 生成的全部 11 类输出（共 12 个文件：`pan.fa`、`offsets.tsv`、`ref_to_pan.chain`、`sv.tsv`、`genotypes.tsv`、`inserts.tsv`、`ref.lifted.gff3`、`centromere.tsv`、`unresolved.tsv`、`unresolved.fa`、`summary.tsv`、`log`）必须**逐字节一致**。
- **严格保持现有的跨阶段全局 `UNRES_*` 编号与落表顺序契约**（严禁并发竞争分配或跨阶段挪动编号时机）：
  1. **Stage 2.3**：着丝粒 `unresolved` 记录；
  2. **Stage 2.4**：仅 **`UNANCHORED_END`** 按 `frag_idx = 0 .. K-1` 顺序分配 `{chrom}_UNRES_*` 并立即追加入 `unresolved_records`；
  3. **Stage 2.5**：**`REPEAT_LIKE`** 按 `sorted_group_pairs` 顺序先分配 `{chrom}_UNRES_*` 并更新 `seen_exported_intervals`，记录暂存于 `guard_res.unresolved_records`；
  4. **Stage 6a**：**`INV_EDGE` / `INV_CONFLICT`** 按 `(rmin, rmax, cls_name)` 顺序分配后续 `{chrom}_UNRES_*` 并追加入 `unresolved_records`（记录 `chain_unres_idxs`）；
  5. **Stage 6b**：**`LARGE_JUMP`** 按 `(rank_a, rank_b)` 顺序提取非参考段、分配后续 `{chrom}_UNRES_*` 并追加入 `unresolved_records`（记录 `lj_unres_idxs` 供 Stage 7 回填 `region_id` 与 `{chrom}_LARGE_JUMP_k`——**严禁提前到 Stage 2.4 按 `frag_idx` 编号**）；
  6. **Stage 6c & Stage 7**：按 `sorted_group_pairs` 顺序将暂存的 `REPEAT_LIKE` 记录追加入 `unresolved_records`，并由 Stage 7 统一回填 `region_id` 与 `event_id`。
- **消除基于对象内存地址的基因分型判等**：将 Stage 6 中全部 6 处依赖 Python 指针地址的 `id(ca)` 判等（`INV`、`LARGE_JUMP`、`DUP_TRA`、`CNV`、`DEL`）统一改为 `_ContigAnalysis` 上的显式整数标识 `contig_idx: int`，彻底杜绝跨进程对象拷贝导致基因型静默退化为 `0` 或 `.`。

### 2.2 内存安全：IPC 零大数组冗余序列化 + COW 脏页抑制（禁止 OOM）
- **Linux `fork` 零拷贝共享只读图**：使用 `multiprocessing.get_context("fork")`，依靠操作系统写时复制（COW）让所有 worker 共享只读 `CompactNodeStore`（连续 `bytes` + NumPy 数组）、`LazyStepTuple` 与 `CompactRefNodeSet`。
- **严禁隐藏的跨进程大字段 `pickle` 复制**：
  1. **剥离 `LazyProvenanceTuple._node_lens`**：`DeviationSegment.detached()` 及长度 `> 64` 的 `ReverseChainRecord.novel_unresolved_slices` 内部的 `LazyProvenanceTuple._node_lens` 指向全图 `CompactNodeStore._node_lens`（`int32[max_nid + 1]`，单份达 40–200 MB）。Stage 2.4 worker 返回前（或通过自定义 `__getstate__`）必须剥离 `_node_lens`（置空数组），由主进程归并时 O(1) 重新绑定回 `graph._node_lens`，防止 32 条 contig 在主进程反序列化出 32 份独立的全图数组副本（节省 1.5–6 GB 内存）；
  2. **精简 `node_visit_counts`**：Stage 2.4 回传时仅保留紧凑的 `_multi_nids: np.ndarray`（Stage 2.5 仅判断访问次数 `> 1`），不序列化大型 `frozenset[int]` (`_multi_set`) 与未使用的 `_visited_bits`；
  3. **剥离 Stage 2.5 顶层冗余序列**：`_screen_and_solve_group_worker` 返回前除剥离 `aligner`、`exemplar.steps`/`provenance`/`seq` 外，还必须清空顶层从不使用的 `sol.backbone_seq = ""`（避免每次回传最高 500 KB 参考序列），且当 `not config.dump_repeat_sequences` 时清空 `REPEAT_LIKE` `UnresolvedRecord.seq = ""`。
- **COW 缓存预热、只读纪律与 `gc.freeze()` 规范**：
  1. Stage 2.4 `fork()` 前，主进程显式调用 `_get_lazy_backbone_anchor_arrays` 预热 `graph.ref_fragment.steps._anchor_cache`，避免各 worker 重复构建上百 MB 的参考锚点数组；
  2. 每次 `gc.freeze()` 前必须先执行 `gc.collect()`（并调用 `_repeat_mask_intervals_cached.cache_clear()`），避免将待回收垃圾冻入永久代；在 `finally` 中确保清空模块级 `_STAGE*_FORK_CTX` 并恢复 `gc.unfreeze()`；
  3. **进程池运行期严禁写共享对象**：Stage 2.4 子进程严禁执行原串行末尾针对输入 `frag` 的 `is_last_for_src` `_orig_codes` 原地清空（着丝粒断臂的左右两个 `normal_fragment` 共享同一 `_orig_codes`，同一 worker 复用时会写坏兄弟分片）；Stage 2.5 运行期父进程严禁边收批次边原地修改 `terminal_devs_by_pair[pair]`（会写脏共享列表内存页触发 64 进程 COW），所有释放统一放在 `with ProcessPoolExecutor` 退出之后。

### 2.3 `fork` 与原生 C / Python 线程安全边界
- **封堵 `numpy` 底层 BLAS / OpenMP 隐式后台线程**：在 `panflat/__init__.py` 与 `panflat/cli.py` 最顶部（**`import numpy` 之前**）通过 `os.environ.setdefault` 将 `OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`OMP_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 设为 `"1"`，防止动态库加载时自动拉起多达 64 个常驻 C pthread（否则在 Python 3.12 下 `os.fork()` 会刷屏多线程 `DeprecationWarning`、造成 `64 × 64` 线程争用甚至死锁）。
- **子进程锁重置**：在 `panflat/progress.py` 中注册 `os.register_at_fork(after_in_child=...)` 重置 `_PROGRESS_LOCK`，确保子进程进度输出永不继承死锁状态。
- **单向时序红线**：严格保持 **先完成所有 `fork` 多进程阶段（可选 Stage 2.3 → Stage 2.4 → Stage 2.5，每阶段池完全关闭回收），最后在 Stage 3 (`--qc`) 进入 `ThreadPoolExecutor` + `mappy.ThreadBuffer` 多线程阶段**；且绝不对 `"fork"` 模式的 `ProcessPoolExecutor` 配置 `max_tasks_per_child`（防止运行期在 `_ExecutorManagerThread` 线程存活时再次触发 `fork`）。

### 2.4 规范与依赖零变动
- 不修改 `panflat_v1_final_revised.md`，不改变公开 CLI 参数，不引入新第三方依赖（仅用标准库 `multiprocessing` / `concurrent.futures` + 现有 `mappy.ThreadBuffer`）。
- 非 Linux 或不支持 `fork` 的平台自动优雅回退到安全路径。

---

## 3. 分模块改造方案

### 模块 A（P0，收益最高）：`run_qc`（`--qc`）真多线程并行

**目标文件**：`panflat/verify_qc.py`（`run_qc`，L171–L398）

1. **保持内存单线程建索引（严禁改走临时文件索引）**：
   - 经核查上游 `mappy.pyx`（L151–165），传入内存字符串 `seq=...` 时底层调用单线程 C 函数 `mappy_idx_seq`（忽略 `n_threads`），**并硬编码设置 `self.map_opt.mid_occ = 1000`**；而文件索引 `fn_idx_in` 分支使用默认 `mid_occ`（如 `asm20` 下约 50）。
   - 为严格守住 T22 逐字节一致性，**严禁改写临时 FASTA 走 `fn_idx_in`**，保持现有 `mappy.Aligner(seq=...)` 单线程建索引（总耗时仅秒级），将多核并发全部投入占比 >95% 的 `map()` 比对阶段。
2. **C 层无 GIL 多线程并发比对 + 动态负载均衡**：
   - 上游 `cmappy.h::mm_map_aux` 全程位于 `Py_BEGIN_ALLOW_THREADS` 内，`mappy` 在 C 层完全释放 GIL；所有线程零拷贝共享同一只读索引 `self._idx`；
   - 预先将 `piece_seqs` 编码为 ASCII `bytes` 列表（`mappy.pyx` 对 `bytes` 输入直接跳过 GIL 下的 `.encode()`）；
   - 新增辅助函数 `_map_queries_parallel(aligner_main, aligner_short, query_indices, piece_bytes, piece_lens, n_threads)`：
     - 将 `query_indices` 按 `piece_lens` **降序排列**（LPT 优先跑长片段/重片段），由 `ThreadPoolExecutor(max_workers=n_threads)` 启动 `n_threads` 个工作线程；
     - 每个工作线程内部持久创建一对独立的 `buf_main = mappy.ThreadBuffer()` 与 `buf_short = mappy.ThreadBuffer()`，通过线程安全游标每次动态领取小批次（如 4–8 条 query）循环调用 `aligner.map(seq_b, buf=buf)` 并完整消费生成器，彻底消除高拷贝 TE / 超长插入片段造成的长尾木桶效应。
3. **100% 确定性保障**：
   - `mm_tbuf_t` 仅管理线程局部 `kalloc` 内存池，命中集合与线程数及调度顺序 100% 无关；
   - 每个 `idx_q` 独立写入各自槽位 `out[idx_q]`，汇总时经 `_merge_intervals(cov_intervals, piece_len)` 排序求并集；同时兼容 `config is None` 时默认 `qc_threads = 1`。
4. **细粒度实时进度日志**：
   - 通过 `log_progress`（仅写 `stderr`，不改动 `.log` 文件）输出 Stage 1 耗时及 Stage 2 的 `round k / (2 * n_bits)` 实时进度。

---

### 模块 B（P0，核心算法真多核）：Stage 2.5 融合 `repeat-guard` 筛查与平行分支组求解

**目标文件**：`panflat/linearizer.py`（L3454–L3615）

**改造设计**：
1. **消除跨组共享可变状态与 IPC 冗余体积**：
   - 将每个非空组 `pair = (rank_a, rank_b)` 的 `2.5a 候选提取 + repeat-guard + 环折叠` 与 `2.5b solve_parallel_group` 合并为纯函数式任务 `_screen_and_solve_group_worker(batch_pairs)`；
   - worker 内用局部占位列表收集 `REPEAT_LIKE` 未决记录（不在 worker 内触碰全局 `unres_counter_ref` 和 `seen_exported_intervals`）；
   - worker 返回前执行彻底裁剪：剥离 `rep.aligner`、`rep.member_occurrences`、`rep.exemplar` 序列与步、**清空 `sol.backbone_seq = ""`**、裁剪 `guard_res.kept_candidates`，并在 `not config.dump_repeat_sequences` 时将 `REPEAT_LIKE` 记录的 `seq` 置为 `""`。
2. **空组快筛 + LPT 碱基预算动态分包（消除高变簇长尾空转）**：
   - 主进程先对 `sorted_group_pairs` 做 O(1) 快筛：所有 `tdev.branch_provenance` 均为空的纯 `DEL`/`CNV` 组直接在主进程赋空 `GroupRepeatGuardResult()`，不进进程池；
   - 对剩余有效组 `active_pairs`，计算每组总候选碱基量 `group_bp`，按 **`group_bp` 降序（Longest Processing Time first）并按碱基预算动态打包**（例如 `group_bp >= 10_000` 的重型复杂位点单组独立成包，微型位点按上限 32–64 组打包），防止染色体上物理相邻的抗病基因簇 / 近着丝粒大插入扎堆落入同一静态分块导致 1 核忙、63 核等。
3. **Linux `fork` 多进程池 + `gc.collect()` / `gc.freeze()`**：
   - 通过模块级只读上下文 `_STAGE5_FORK_CTX` 共享 `(graph, config, ref_steps, n_ref, top_state_key, terminal_devs_by_pair, claimed_source_intervals)`；
   - 当 `config.threads > 1`、平台支持 `fork`、且达到并行阈值（如 `len(active_pairs) >= 32` 或设置了测试环境变量 `PANFLAT_FORCE_FORK=1`）时：先执行 `_repeat_mask_intervals_cached.cache_clear()`、`gc.collect()`、`gc.freeze()`，再启动 `ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("fork"))`；
   - 按批次完成度在主进程定期刷新 `stderr` 进度；`finally` 中确保 `gc.unfreeze()` 并清空 `_STAGE5_FORK_CTX = None`。
4. **池退出后的主进程确定性归并与内存回收**：
   - 进程池完全退出后，主进程严格按原始 `sorted_group_pairs` 顺序遍历结果，统一分配 `record_id = f"{chrom}_UNRES_{unres_counter_ref[0]}"` 与 `exported_fasta_id = seen_exported_intervals.setdefault(key, rec_id)`，写入 `group_guard_by_pair` / `group_solutions`；
   - 最后在主进程统一清空 `devs_at_pair` 的 `branch_provenance` 及 `contig_analyses` 的 `node_visit_counts`。

---

### 模块 C（P1，大染色体多单倍型加速）：Stage 2.4 分支 contig 倒位归一化与加权 LIS 并行

**目标文件**：`panflat/linearizer.py`（L2865–L3452）与 `panflat/models.py`

**改造设计**：
1. **单 contig 纯函数化与零冗余序列化**：
   - 给 `_ContigAnalysis` 增加显式字段 `contig_idx: int`，替换 Stage 6 全部 `id(ca)` 指针比较；
   - 抽出 `_analyze_single_normal_fragment(frag_idx)`，只读访问 `_STAGE4_FORK_CTX`（含 `graph`, `config`, `normal_fragments_sorted`, `ref_steps`, `n_ref`, `top_state_key`, `claimed_source_intervals`），**严禁修改输入 `frag` 的 `_codes` / `_orig_codes`**；
   - 返回真正轻量的 `_FragmentStage4Result`：
     - `analysis: _ContigAnalysis`（其中 `node_visit_counts` 仅携带紧凑 `_multi_nids: np.ndarray`）；
     - `inv_chains: list[ReverseChainRecord]`（已剥离内部 `LazyProvenanceTuple._node_lens` 引用）；
     - `terminal_devs: list[tuple[tuple[int, int], DeviationSegment]]`（已完成微变异过滤、`.detached(keep_provenance=keep_prov)` 并剥离 `_node_lens` 引用）；
     - `large_jump_devs: list[tuple[tuple[int, int], DeviationSegment]]`（已 `.detached()` 并剥离 `_node_lens` 引用）；
     - `pending_unanchored_unresolved: list[tuple[StepProvenance, str]]`（**仅含 `UNANCHORED_END`**；`LARGE_JUMP` 严格保留在 Stage 6b 提取编号）。
2. **缓存预热 + Linux `fork` 多进程并行 + 确定性归并**：
   - `fork` 前在主进程预热 `ref_steps._anchor_cache`，执行 `gc.collect()` + `gc.freeze()`；
   - 当 `config.threads > 1`、平台支持 `fork` 且 `len(normal_fragments_sorted) >= 2` 且（`len(ref_steps) >= 5_000` 或 `PANFLAT_FORCE_FORK=1`）时，启动 `n_workers = min(max_fork_workers, len(normal_fragments_sorted))` 个 `fork` 进程并行处理各 contig；
   - 池退出并 `gc.unfreeze()` 后，主进程严格按 `frag_idx = 0 .. K-1` 升序：
     1. 将所有反序列化的 `LazyProvenanceTuple._node_lens` 重新绑定回 `graph._node_lens`，重建紧凑 `CompactVisitCounts`；
     2. 依次归并 `contig_analyses`、`inv_chains_by_key`、`terminal_devs_by_pair`、`large_jump_devs_by_pair`；
     3. 对 `pending_unanchored_unresolved` 按原顺序调用 `_emit_normal_unresolved(..., reason="UNANCHORED_END")` 分配 `UNRES_*` ID；
     4. 立即释放 `normal_fragments_sorted` 内存。

---

### 模块 D（P2，可选）：着丝粒 `extract-islands` 模式 `fork` 并行加速

**目标文件**：`panflat/centromere.py`（`process_centromere_regions`，L1109–L2050）

- 鉴于 `process_centromere_regions` 处于 **Stage 2.3（早于 Stage 2.4/2.5 `fork`）**，且 `extract-islands` 的 Step C（15-mer 滑窗屏蔽 + 节点切片）受 Python GIL 限制，**不在 Stage 2.3 使用 `ThreadPoolExecutor`**，以严格遵守第 2.3 节“全程先 `fork`、仅最后 Stage 3 多线程”的边界。
- 若需加速 `--centromere-mode extract-islands`：在单个着丝粒区域内先在父进程构建只读 `region_aligner`，用 `fork` 进程池按单倍型并行执行候选提取 + Step B（`subtract_reference_intervals_mappy`）+ Step C（`repeat_mask_intervals` + 切片），主进程按 `graph.all_haplotypes` 确定顺序归并统计指标、分配 `{region_id}_UNRES_*` 并完成 Step D 岛聚类。
- 若日常主要使用 `--centromere-mode bypass`（毫秒级完成），此模块可留作可选增强。

---

### 明确暂不并行的部分（及理由）

| 阶段 | 暂不并行的原因 |
|---|---|
| `gfa_io.py` GFA 流式加载 | 顺序字节流 + SHA-256 哈希，受单盘 I/O 与逐行解码支配；强行切块会破坏流式低内存设计。仅补充关键节点（S/L 解析完成、P/W 解析完成、FASTA 校验完成）的 `stderr` 进度日志 |
| `verify_qc.py::verify_build_outputs` | 顺序校验输出文件与 SHA-256，内存占用低、代码复杂度敏感，保持单线程以保证不变量审计可靠性 |

---

## 4. 风险与防御清单

| 风险 | 影响 | 防御措施 |
|---|---|---|
| **IPC 反序列化大数组膨胀** | `LazyProvenanceTuple._node_lens` 跨进程复制 32 份全图数组耗尽内存 | 子进程返回前剥离 `_node_lens`、顶层清空 `sol.backbone_seq`、精简 `node_visit_counts`、按需清空未导出 `REPEAT_LIKE.seq`；主进程归并时 O(1) 重绑 `graph._node_lens` |
| **COW 内存放大** | 多进程 GC 扫描或运行期改写共享容器造成物理页分裂 | 1) 预热 `ref_steps._anchor_cache`；2) `gc.collect()` 后 `gc.freeze()`；3) 运行期严禁原地修改输入 `frag` 或 `terminal_devs_by_pair`，池退出后再统一清理；4) 支持可配置的 `max_fork_workers` 上限 |
| **`numpy` OpenBLAS 后台线程 + `fork` 警告/死锁** | Python 3.12 刷屏 `DeprecationWarning`、子进程锁死或 `64 × 64` 线程争用 | 在 `panflat/__init__.py` 与 `panflat/cli.py` 导入 `numpy` 前默认设置 `OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`，并在 `after_in_child` 重置 `_PROGRESS_LOCK`；严禁对 `fork` 池启用 `max_tasks_per_child` |
| **T22 串行/并行不一致或 `event_id` 漏回填** | 违反规格 §6 验收红线或 `panflat verify` 报错 | 1) 严格保留 `UNANCHORED_END` → `REPEAT_LIKE` → `INV_EDGE`/`INV_CONFLICT` → `LARGE_JUMP` 的全局编号与落表顺序；2) 用显式 `contig_idx` 替换全部 `id(ca)`；3) `run_qc` 保持 `seq=...`（`mid_occ=1000`）内存索引不变 |
| **复杂高变位点长尾空转** | 1 个重型位点包拖慢整轮 64 核等待 | Stage 2.5 先过滤空组，再按候选总碱基量降序（LPT）动态分包；`run_qc` 按片段长度降序配合线程动态游标拉取 |
| **小图测试绕过 `fork` 路径导致回归盲区** | 自适应阈值使常规小测试走回单进程 | 生产默认按规模阈值免除小图 `fork` 开销；同时支持 `PANFLAT_FORCE_FORK=1` 在测试中强制复杂对抗小图完整走通 `fork` + IPC 归并路径 |
| **非 Linux 兼容** | macOS / Windows 无 `fork` | 检测 `"fork" in multiprocessing.get_all_start_methods()`，不可用时自动优雅回退 |

---

## 5. 验证与验收标准

改造完成后必须全部通过以下门禁才合并推送：

1. **静态检查与规范门禁**：
   - `ruff check --select F,E9,W panflat tests`：0 错误
   - `! rg -n '^\s*(import|from)\s+panflat\b' tests/e2e`：0 违规
   - `git diff --exit-code panflat_v1_final_revised.md`：0 字节改动
2. **全量功能与不变量回归**：
   - 现有 **`523 / 523`** 测试（`170` 个单元测试 + `353` 个黑盒 CLI E2E 测试）全部通过。
3. **新增真多核与 T22 强确定性测试**：
   - **对抗边角强制 `fork` 测试**（`PANFLAT_FORCE_FORK=1`）：覆盖普通区（含 `INV`、`INV_EDGE`、`INV_CONFLICT`、`CNV`、`DEL`、`DUP_TRA`、`LARGE_JUMP`、`UNANCHORED_END`、大片段 repeat-guard 切分残余、多分支递归聚类）+ 着丝粒断臂共享 `_orig_codes`（`bypass` 与 `extract-islands`）+ `--dump-repeat-sequences` + `--ref-gff` + `--qc`；
   - **真实阈值触发测试**：无需环境变量，直接满足 `ref_steps >= 5,000` 且非空 `terminal_groups >= 32` 的多单倍型测试图；
   - 分别以 `--threads 1` 和 `--threads 4` 运行 `panflat build` 及独立 `panflat verify`，断言全部 12 个输出文件（`.pan.fa`、`.offsets.tsv`、`.ref_to_pan.chain`、`.sv.tsv`、`.genotypes.tsv`、`.inserts.tsv`、`.ref.lifted.gff3`、`.centromere.tsv`、`.unresolved.tsv`、`.unresolved.fa`、`.summary.tsv`、`.log`）**逐字节完全相同（SHA-256 一致）**，且 `stderr` 无任何 Python 3.12 多线程 `fork` 警告。
4. **多核加速与内存实测基准**：
   - 合成 30+ 单倍型染色体规模图（含大量插入片段与 `--qc`），记录 `threads=1` vs `threads=N` 的各阶段耗时、CPU 利用率和峰值内存（RSS），确认真实多核加速且无 IPC/COW 内存异常放大。

---

## 6. 默认落地策略（可按需微调）

1. **实施范围**：默认一次性完成 **P0（模块 A `--qc` 多线程 + 模块 B Stage 2.5 `fork` 多进程）+ P1（模块 C Stage 2.4 按 contig `fork` 多进程）+ `gfa_io.py` 细粒度 `stderr` 进度日志**（P2 `extract-islands` `fork` 并行可按需一并纳入）。
2. **`fork` 进程数上限策略**：`--qc` 多线程阶段直接按 `config.threads`（如 `64`）并发；`fork` 多进程阶段（Stage 2.4 / 2.5）默认按 `min(config.threads, 任务数)` 并发（同时支持环境变量 `PANFLAT_MAX_FORK_WORKERS` 在中小内存机器上手工限制进程池上限）。
