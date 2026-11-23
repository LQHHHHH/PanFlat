# PanFlat v1 真多核并行改造计划（`MULTICORE_PLAN.md`）

> **目的**：解决 `panflat build --threads N`（如 `-t 64`）在真实染色体多单倍型数据（例如 33 单倍型梨属 Chr15A）上长期只跑单核（~100% CPU）的问题，在**严格保证串行与并行输出逐字节一致（规格 T22）**且**不引发内存爆炸**的前提下，让核心计算阶段真正吃满多核。
>
> **状态**：待审阅确认（本文档仅作方案评审，尚未改动核心并行代码）。

---

## 1. 现状瓶颈诊断

逐一审计当前代码库后，`panflat build` 全流程各阶段的并行现状如下：

| 阶段 | 代码位置 | 当前状态 | 为什么跑单核 / 瓶颈分析 |
|---|---|---|---|
| **1. 输入加载与预检** | `panflat/gfa_io.py::load_and_validate_inputs` | 单核 | 顺序流式读盘、算 SHA-256、构建紧凑 NumPy 存储；主要受磁盘 I/O 与解析带宽限制 |
| **2.1–2.3 着丝粒定界与处理** | `panflat/centromere.py::process_centromere_regions` | 单核 | `bypass` 很快；`extract-islands` 时 Step B（`mappy` 减法）和 Step C（滑窗重复屏蔽）单核串行 |
| **2.4 倒位归一化 / 加权 LIS / 偏离段提取** | `panflat/linearizer.py` (L2865–L3452) | **单核** | 对 32 条分支 contig 逐条循环执行归一化、投影、LIS、微变异过滤和局部重锚，**彼此完全独立但未并行** |
| **2.5a 通用 repeat-guard 滑窗筛查** | `panflat/linearizer.py` (L3508–L3570) | **单核** | 对全部 `terminal_groups` 逐个做候选提取、15-mer 滑窗屏蔽与环折叠，完全串行 |
| **2.5b 平行分支组递归与聚类** | `panflat/linearizer.py` (L3573–L3615) | **伪多核（GIL）** | 整个项目**唯一**使用 `config.threads` 的地方，但用的是 Python `ThreadPoolExecutor`；受 Python 3.12 **GIL** 限制，纯 Python 递归/并查集/去重/窗口校验同一时刻只能跑 1 核 |
| **3. 写出与 `--qc` 冗余分析** | `panflat/verify_qc.py::run_qc` (L171–L398) | **完全单核** | `--qc` 做 Stage 1（全部 ≥150 bp 插入片回比整条参考）+ Stage 2（`2 × ceil(log2(N))` 轮 `mappy.Aligner` 建索引与两两互比），**既未传 `n_threads` 也未并发 `map()`**，在大染色体上极易长时间吃满单核 |
| **4. 最终不变量校验** | `panflat/verify_qc.py::verify_build_outputs` | 单核 | 流式逐块核对 Invariant I/II/III 和 GFA SHA-256；顺序 I/O 为主 |

**结论**：真实数据跑 `--threads 64 --qc` 时，CPU 长期停在 100%（单核）主要由三大计算瓶颈造成：**Stage 3 (`--qc`)**、**Stage 2.5 (repeat-guard + 分支组求解)**、**Stage 2.4 (多单倍型倒位与 LIS)**。

---

## 2. 核心设计约束（红线）

1. **T22 严格确定性（不可妥协）**：
   - `--threads 1` 与 `--threads N` 生成的全部 11 个输出文件（`pan.fa`、`offsets.tsv`、`ref_to_pan.chain`、`sv.tsv`、`genotypes.tsv`、`inserts.tsv`、`ref.lifted.gff3`、`centromere.tsv`、`unresolved.tsv`、`unresolved.fa`、`summary.tsv`、`log`）必须**逐字节一致**。
   - 任何共享自增编号（如 `{chrom}_UNRES_1, 2, ...`）与去重映射（`seen_exported_intervals`）禁止在并发 worker 内竞争分配，一律改由主进程按确定顺序统一归并编号。
2. **内存安全（禁止 OOM）**：
   - 不向子进程 pickle 复制整个 `GFAGraph`（几十 MB~数 GB 的节点序列和参考索引）。
   - Linux 下使用 **`multiprocessing.get_context("fork")`**，依靠操作系统写时复制（COW）让所有 worker 共享只读的 `CompactNodeStore`（连续 `bytes` + NumPy 数组）；在 `fork()` 前执行 `gc.freeze()`、池关闭后 `gc.unfreeze()`，防止子进程 GC 扫描修改父进程对象头触发大量 COW 页复制。
3. **Fork 与多线程安全边界**：
   - **严禁**在已创建后台 C 线程或活跃 `ThreadPoolExecutor` 之后再调用 `fork()`（避免 POSIX fork 死锁）。
   - 严格保持执行时序：**先完成所有 `fork` 多进程阶段（Stage 2.4 → Stage 2.5），全部回收子进程后，再进入 Stage 3 (`--qc`) 的 `ThreadPoolExecutor` + `mappy` 多线程阶段**。
4. **规范与依赖零变动**：
   - 不修改 `panflat_v1_final_revised.md`，不引入新第三方依赖（仅用标准库 `multiprocessing` / `concurrent.futures` + 现有 `mappy.ThreadBuffer`）。
   - 非 Linux 或不支持 `fork` 的平台自动优雅回退到现有路径。

---

## 3. 分模块改造方案

### 模块 A（P0，收益最高）：`run_qc`（`--qc`）真多线程并行

**目标文件**：`panflat/verify_qc.py`（`run_qc`，L171–L398）

1. **多线程建索引**：
   - `mappy.Aligner(seq=..., preset="asm20", best_n=50, n_threads=idx_threads)` 和 `aligner_short` 显式传入 `n_threads`（限制 `idx_threads = min(qc_threads, 16)`，避免小索引开 64 线程的调度抖动）。
2. **多线程并发比对（C 层释放 GIL，零索引拷贝）**：
   - 新增辅助函数 `_map_queries_parallel(aligner_main, aligner_short, query_indices, piece_seqs, piece_lens, n_threads)`：
     - 将 `query_indices` 按块切分给 `ThreadPoolExecutor(max_workers=n_threads)`；
     - 每个工作线程内部创建独立的 `buf_main = mappy.ThreadBuffer()`（及 `buf_short = mappy.ThreadBuffer()`），调用 `aligner.map(seq, buf=buf)`；
     - `mappy` 在 C 层（minimap2）完全释放 GIL，且所有线程共享同一份只读索引，**几乎零额外内存**，可稳定利用多核。
3. **确定性保障**：
   - 每个 `idx_q` 独立收集命中列表，汇总时调用 `_merge_intervals(cov_intervals, piece_len)`（先排序再取区间并集），结果与线程调度顺序完全无关；
   - 实施前先用脚本实测验证 `mappy.Aligner(n_threads=1)` vs `n_threads=K` + `ThreadBuffer` 在相同目标序列上命中结果 100% 一致（若发现 `n_threads > 1` 建索引存在任何微小次优命中顺序差异，则索引固定单线程构建、仅对 `map()` 做全线程并行，确保 T22 绝对逐字节一致）。
4. **细粒度实时进度日志**：
   - 输出 Stage 1 耗时，以及 Stage 2 的 `round k / (2 * n_bits)` 实时进度，不再黑盒等待。

---

### 模块 B（P0，核心算法真多核）：Stage 2.5 融合 `repeat-guard` 筛查与平行分支组求解

**目标文件**：`panflat/linearizer.py`（L3454–L3615）

**当前问题**：
- Stage 2.5a（`extract_candidate_blocks` + `screen_and_fold_candidate` 含 15-mer 滑窗屏蔽与环折叠）完全单核；
- Stage 2.5b（`solve_parallel_group` 含组件递归、`edlib`/`mappy` 4 判据聚类）用 `ThreadPoolExecutor`，受 Python GIL 限制几乎只跑 1 核。

**改造设计**：
1. **消除跨组共享可变状态（保证确定性）**：
   - 将每个组 `pair = (rank_a, rank_b)` 的 `2.5a 候选提取 + repeat-guard + 环折叠` 与 `2.5b solve_parallel_group` 合并为纯函数式单组任务 `_screen_and_solve_group_worker(pair)`；
   - worker 内不再直接写入全局 `unres_counter_ref` 和 `seen_exported_intervals`，而是返回该组原始顺序的 `pending_unresolved` 列表（`record_id` / `exported_fasta_id` 留空占位）；
   - worker 返回前保持现有的轻量化裁剪（剥离 `aligner`、`exemplar.steps`、`seq` 等大字段），将 IPC pickle 体积降到极小。
2. **Linux `fork` 多进程池 + `gc.freeze()` + 分块调度**：
   - 通过模块级只读上下文 `_STAGE5_FORK_CTX` 保存 `(graph, config, ref_steps, n_ref, top_state_key, terminal_devs_by_pair, claimed_source_intervals, contig_visit_counts_by_idx)`；
   - 仅当 `config.threads > 1`、平台支持 `fork`、且任务规模达到并行阈值（如 `len(sorted_group_pairs) >= 32`）时启动 `ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("fork"))`；
   - **Worker 数上限与分块策略**：
     - 为了避免 64 个进程在简单小位点上产生过多 IPC 和 COW 脏页，将 `sorted_group_pairs` 按块分发（例如每块 `batch_size = max(1, min(64, n_groups // (n_workers * 8)))`），并按确定顺序收集所有批次结果；
     - 用 `gc.freeze()` 冻结预分配只读对象后 `fork()`，池退出 `finally` 中 `gc.unfreeze()` 并清空 `_STAGE5_FORK_CTX` 释放引用；
     - 按每批完成回报实时打印 `stderr` 进度（每 ≥5s 或完成时刷新一次）。
3. **主进程确定性归并（O(U) 毫秒级）**：
   - 严格按 `sorted_group_pairs` 顺序遍历各组返回结果，在主进程统一补填 `record_id = f"{chrom}_UNRES_{unres_counter_ref[0]}"` 和 `exported_fasta_id = seen_exported_intervals.setdefault(key, rec_id)`，并写入 `group_guard_by_pair` / `group_solutions`；
   - 同步把 `devs_at_pair` 中不再需要的 `branch_provenance` 清空释放内存。

---

### 模块 C（P1，大染色体多单倍型加速）：Stage 2.4 分支 contig 倒位归一化与加权 LIS 并行

**目标文件**：`panflat/linearizer.py`（L2865–L3452）

**当前问题**：
- 32 条（或更多）alt contig 逐条执行 `normalize_inversions_top_level` → `project_steps_onto_backbone` → `compute_weighted_lis` → `exact_ref_match_mask` → `compute_raw_visit_counts` → 偏离段提取与局部重锚，完全单核。

**改造设计**：
1. **单 contig 纯函数化**：
   - 抽出 `_analyze_single_normal_fragment(frag_idx)`，读取只读的 `_STAGE4_FORK_CTX`（含 `graph`, `config`, `normal_fragments_sorted`, `ref_steps`, `claimed_source_intervals`）；
   - 返回紧凑的 `_FragmentStage4Result`：
     - `analysis: _ContigAnalysis`
     - `inv_chains: list[ReverseChainRecord]`
     - `terminal_devs: list[tuple[tuple[int, int], DeviationSegment]]`（已做完微变异过滤并 `.detached(keep_provenance=keep_prov)`，体积小且完全可 pickle）
     - `large_jump_devs: list[tuple[tuple[int, int], DeviationSegment]]`
     - `pending_unresolved: list[tuple[str, str, StepProvenance, str, str, bool]]`（`UNANCHORED_END` / `LARGE_JUMP` 未编号原记录）
2. **Linux `fork` 多进程并行 + 确定性归并**：
   - 当 `config.threads > 1` 且 `len(normal_fragments_sorted) >= 2` 且 `len(ref_steps) >= 5_000`（小图免 fork 开销）时，开 `n_workers = min(config.threads, len(normal_fragments_sorted))` 个 `fork` 进程并行处理所有分支 contig；
   - 主进程严格按 `frag_idx = 0 .. K-1` 顺序归并 `contig_analyses`、`inv_chains_by_key`、`terminal_devs_by_pair`、`large_jump_devs_by_pair`，并按原顺序调用 `_emit_normal_unresolved` 分配 `UNRES_*` ID；
   - 每完成一批 contig 实时打印 `stderr` 进度，归并后立刻释放 `normal_fragments_sorted` 内存。
3. **单核与多核等价性**：
   - 各 contig 在 Stage 2.4 之间**零数据依赖**，唯一跨 contig 状态就是 `_emit_normal_unresolved` 的顺序编号——只要主进程按 `frag_idx` 升序补发编号，数学上 100% 等价于串行循环。

---

### 模块 D（P2，可选）：着丝粒 `extract-islands` 模式并行加速

**目标文件**：`panflat/centromere.py`（`process_centromere_regions`，L1109–L2050）与 `panflat/repeat_guard.py`（`subtract_reference_intervals_mappy`）

- 规格 §6 已明确说明：“特殊区间可按区域并行；一个区域内复用同一参考索引，处理结果统一稳定排序；不复制整个 GFA 到每条候选任务。”
- 如果后续你需要跑 `--centromere-mode extract-islands`：
  - 给 `subtract_reference_intervals_mappy` 传入可选 `buf: mappy.ThreadBuffer | None`，在单个着丝粒区域内复用同一份 `mappy.Aligner`，用 `ThreadPoolExecutor` 并发对多个单倍型的候选块做 Step B 参考减法；
  - 若当前主要用 `--centromere-mode bypass`，此模块几乎不耗时，可放到 P0+P1 落地后再做（或一并纳入）。

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
| **COW 内存放大**（64 进程同时跑 Python 造成内存高于单核） | 可能导致大染色体 OOM | 1) `fork()` 前 `gc.freeze()`；2) `CompactNodeStore` 与 NumPy 数组本就无逐元素 Python refcount；3) 为 `fork` 多进程池提供合理的默认并发上限建议（如 Stage 4 上限 `min(threads, n_contigs)`，Stage 5 对大内存图可通过参数或内存估算控制，必要时限制 `max_fork_workers`） |
| **fork + 线程死锁** | POSIX 下若父进程已有其他线程活跃再 `fork()` 可能锁死 | 严格保持顺序：**Stage 2.4 (`fork`) → 池关闭回收 → Stage 2.5 (`fork`) → 池关闭回收 → Stage 3 `--qc` (`ThreadPoolExecutor`)**；绝不混用 |
| **T22 串行/并行输出不一致** | 违反规格 §6 验收红线 | 1) 所有 `UNRES_*` 编号和 `seen_exported_intervals` 全部延后到主进程按确定键顺序生成；2) `run_qc` 若 `mappy` 多线程建索引有任何非确定性，立刻固定单线程建索引、仅并发 `map()`；3) 写专门的 SHA-256 逐字节对比测试 |
| **小图 / 单元测试变慢** | 525 个小单元测试频繁 `fork` 会拖慢测试 | 设置自适应并行阈值（小图自动走单进程），并额外编写可显式强制触发 `fork` 路径的回归测试 |
| **非 Linux 兼容** | macOS / Windows 无 `fork` | 检测 `"fork" in multiprocessing.get_all_start_methods()`，不可用时无缝回退到现有 `ThreadPoolExecutor` / 单进程 |

---

## 5. 验证与验收标准

改造完成后必须全部通过以下门禁才合并推送：

1. **静态检查与规范门禁**：
   - `ruff check --select F,E9,W panflat tests`：0 错误
   - `! rg -n '^\s*(import|from)\s+panflat\b' tests/e2e`：0 违规
   - `git diff --exit-code panflat_v1_final_revised.md`：0 字节改动
2. **全量功能与不变量回归**：
   - 现有 `525 / 525` 单元测试 + 黑盒 E2E 测试全部通过。
3. **新增真多核与 T22 强确定性测试**：
   - 构造覆盖普通区（含倒位、CNV、大片段 repeat-guard 屏蔽、多分支递归聚类）+ 着丝粒（`bypass` 与 `extract-islands`）+ `--ref-gff` + `--qc` 的多单倍型测试集；
   - 分别以 `--threads 1` 和 `--threads 4`（强制触发 Stage 2.4 `fork`、Stage 2.5 `fork`、Stage 3 `--qc` 多线程）运行 `panflat build`；
   - 断言全部 11 个输出文件（含 `.pan.fa`、`.sv.tsv`、`.genotypes.tsv`、`.inserts.tsv`、`.unresolved.tsv`、`.unresolved.fa`、`.summary.tsv`、`.log` 等）**逐字节完全相同（SHA-256 一致）**。
4. **多核加速实测基准**：
   - 合成 30 单倍型染色体规模图（含大量插入片段与 `--qc`），记录 `threads=1` vs `threads=N` 的各阶段耗时、CPU 利用率和峰值内存（RSS），确认真实多核加速且内存可控。

---

## 6. 待你确认的 3 个点

审阅后你可以直接回复确认或调整：

1. **实施范围**：是否按 **P0（`--qc` 多线程 + Stage 2.5 `fork` 多进程）+ P1（Stage 2.4 按 contig `fork` 多进程）** 一次性改完？（如果后续也要跑 `--centromere-mode extract-islands`，我也可以顺手把 P2 一起做了。）
2. **`fork` 进程数上限策略**：当你传 `--threads 64` 时，线程型阶段（`--qc`）可直接开到 `64`；对于 `fork` 多进程阶段（Stage 2.4 / 2.5），你是希望直接按 `--threads`（最多 `min(threads, 任务数)`，吃满 64 核），还是希望默认设一个保守上限（比如 32）防止高内存染色体占太多内存？（取决于你服务器总内存有多大。）
3. **小细节**：是否同时在 `gfa_io.py` 补上 GFA 加载的细粒度进度日志（读完 S/L、读完 P/W、FASTA 核对完成）？
