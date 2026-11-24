# PanFlat v1 代码审查问题核查与修复方案（`REVIEW_FIX_PLAN.md`）

> **目的**：针对外部代码审查提出的 5 项问题（P1 × 3、P2 × 2），逐项完成源码级白盒复核与设计规格（`panflat_v1_final_revised.md`）对照，明确区分**真实实现偏差**与**不可误改的原设计机制**，给出不破坏任何既有功能与全局不变量的精确修复方案。
>
> **当前状态**：已完成全部 5 项问题的根因定位、边界推演与兼容性核查；本文档为待执行的修改设计说明（尚未修改业务源码与 `panflat_v1_final_revised.md`）。

---

## 1. 审查意见核查结论与设计影响总览

| 编号 | 级别 | 审查问题简述 | 代码核查结论 | 对原设计的影响与处理原则 |
|---|---|---|---|---|
| **1** | **P1** | `CNV` 把分散节点算成串联拷贝（`Ref: A–U–U–D–B`，`Alt: A–U–X–U–X–U–B` 输出 `k_ref=2, k_alt=3`） | **完全属实**（`linearizer.py` L4618 / L4720） | 实现在分支侧过滤掉不共享节点 `X` 后丢失了原始相邻关系，违反规格 §3.4.3–3.4.4（T17）。修复时补充**分支侧共享步连续性约束**，同时保留侧翼伴随 `DEL`/`INS` 的真实串联 `CNV` 检测能力 |
| **2** | **P1** | `INS`（替换位点删 `D` 的样本 `A–B`）和 `INV`（仅含内部 `B⁺` 缺两侧锚点的短 contig）在证据不足时被填为 `0` | **完全属实**（`linearizer.py` L4952 / L3641） | 直接违反规格 §5.2 取值表，虚高 `n_called`；尤其 `INV` 还会与 §3.3.0 整条 contig 定向叠加，把倒位携带者的内部断裂 contig 误判为 `0`。严格按 §5.2 收紧 `0` 的判定条件即可 |
| **3a** | **P1** | 相似聚类 `200=30I200=` 的 50 bp 滑窗分母未计入纯插入列，合并后未保留 30 bp 序列 | **现象属实，但属原设计预期，严禁照改**（`repeat_guard.py` L966） | `30I` 长度 `< min-sv-len (50 bp)`，按规格 §0、§3.6.3 判据 3、**T5** 与 **T13** 本就应吸收抹平。若把纯插入碱基计入 50 bp 窗口分母，所有 **16–49 bp** 纯小插入都会因 `(50 - L)/50 < 0.70` 被判据 4 否决，直接击穿判据 3 与 **T5** |
| **3b** | **P1** | 200 bp 候选在 400 bp 代表内有两处完全匹配，短序列 `edlib` 分支仍合并为同一等位 | **完全属实**（`repeat_guard.py` L872） | `≤ 1000 bp` 分支调用 `edlib` `mode="NW"` 时漏掉了规格 §3.6.3 判据 1（唯一同向比对：多重命中、拆分比对或无命中时不合并）。需在短序列分支补齐多命中、反向竞争命中与大跨度内部拆分拦截 |
| **4** | **P2** | 从 `A–D–X–B–D` 提取 `X` 后输出 `A–X` 首块仍标 `source_contiguous`；且篡改环折叠 `synthetic` 接头后独立 `verify` 仍放行 | **完全属实**（`linearizer.py` L1303、`verify_qc.py` L2379） | 违反规格 §1 不变量 II 与 §3.6.5。需在构建端按 `pan.fa` 实际紧邻前驱在源路径上是否真实连续来区分首块 `source_contiguous` vs `synthetic`（保留正常纯插入的 `source_contiguous`），并在 `verify` 中补齐源坐标与前驱步连续性校验 |
| **5** | **P2** | 深层嵌套局部重锚（如 1,050 层）触发 Python `RecursionError` | **完全属实**（`linearizer.py` L481） | 规格 §3.5 明确规定通过度量严格递减终止且**“不设 max-depth 参数”**。将 `reanchor_deviation` 改为显式栈（迭代 DFS）即可彻底消除调用栈上限与每层 `frozenset` 拷贝开销，输出逐字节不变 |

**总体判断**：上述问题均**不动摇 PanFlat v1 的总体架构**（路径相对递归线性化、右锚点统一挂载、三大全局不变量、GFF3 确定性平移、着丝粒两档分流均完全自洽）。只要守住下述设计红线，即可在不伤及任何原有特性的前提下一次性闭环修复 **1、2、3b、4、5**（并对 **3a** 做精细边界加固与文档说明）。

---

## 2. 核心设计红线（修复约束）

1. **规格文档零改动**：`panflat_v1_final_revised.md` 保持 0 字节改动（`git diff --exit-code panflat_v1_final_revised.md`）。
2. **严禁破坏 `< min-sv-len` 微小变异吸收规则（T5 / 判据 3）**：
   - 纯小插入（`1–49 bp`，含末端未比对悬垂与内部插入之和 `< min_sv_len`）和纯小缺失（`< min_sv_len`）必须继续允许并入代表序列，绝不能因修改判据 4 滑窗分母把纯小插入阈值隐式压缩到 `≤ 15 bp`。
3. **保留同位点侧翼 `DEL` / `INS` 共存下的串联 `CNV` 检测**：
   - 当参考为 `A–D–U–U–B`、分支为 `A–U–U–U–B` 或 `A–U–U–U–X–B`（即共享重复单元本身在参考与分支上**各自连续成串**，仅外侧伴随缺失或新增插入）时，仍正常输出共存的 `DEL`、`CNV` 与 `INS` 事件并共享 `region_id`。
4. **保留真实相邻纯插入首块的 `source_contiguous` 语义**：
   - 当 `Ref = A–B`、`Alt = A–X–B`（含正向路径、**T23** 整条反向互补 contig、**T7** 倒位归一化内部纯插入），且 `X` 在 `pan.fa` 中无前导 `spacer`、在源路径中确实紧邻左锚点 `A` 时，首个 piece 的 `join_before` 仍须为 `"source_contiguous"`，不得一刀切改为 `"synthetic"`。
5. **T22 严格确定性与零新增依赖**：
   - `--threads 1` 与 `--threads N` 的全部 12 个输出文件必须逐字节一致（SHA-256 完全相同）；不引入任何新第三方依赖。

---

## 3. 逐项根因分析与具体修改方案

### 修改项 1（P1）：`CNV` 局部共享节点周期检测强制保持分支相邻关系

- **目标文件**：`panflat/linearizer.py`（Stage 6c `(ii)`，紧凑路径 L4607–4637，普通路径 L4703–4741）
- **根因定位**：
  - 全区间 `detect_canonical_cnv(b_slice_steps, dev.branch_steps, ...)`（L4447）本身直接在未过滤路径上运行，不存在此问题；
  - 漏洞出在后续的局部共享子串分支（`partial_cnv_intervals`）：代码先验证了共享节点在参考切片 `b_slice_steps` 上的下标集合 `sh_rel` 满足 `sh_rel[-1] - sh_rel[0] + 1 == n_sh`（参考侧连续），但在提取分支侧序列时直接写了：
    - 紧凑路径（L4618–4620）：`br_is_sh = np.isin(br_codes_abs, shared_int); br_sh_lazy = LazyStepTuple(dev.branch_steps._codes[br_is_sh])`
    - 普通路径（L4720–4725）：`br_sh_steps = [st for st in dev.branch_steps if _is_full_step(st) and st.node_id in shared_nids]`
  - 当分支为 `U–X–U–X–U` 时，不共享的 `X` 被静默抽走，原本被 `X` 隔开的 3 个分散 `U` 被拼成了伪连续序列 `U–U–U`，触发 `k_ref=2, k_alt=3` 假阳性。
- **具体改法**：
  1. **紧凑路径（`LazyStepTuple`）**：求出 `br_sh_rel = np.flatnonzero(br_is_sh)` 与 `n_br_sh = int(br_sh_rel.shape[0])`，仅当 `n_br_sh > 0 and int(br_sh_rel[-1]) - int(br_sh_rel[0]) + 1 == n_br_sh`（即所有共享节点在原始归一化分支路径上构成**唯一一段连续无间断切片**，中间无任何非共享节点）时，才取连续切片 `dev.branch_steps._codes[int(br_sh_rel[0]) : int(br_sh_rel[-1]) + 1]` 调用 `detect_canonical_cnv`。
  2. **普通路径（`Sequence[Step]`）**：记录分支侧满足 `_is_full_step(st) and st.node_id in shared_nids` 的原始下标列表 `br_shared_indices`；仅当 `br_shared_indices` 非空且 `br_shared_indices[-1] - br_shared_indices[0] + 1 == len(br_shared_indices)`（中间既无非共享节点，也无 `repeat-guard` 产生的半截切片 `is_partial` step）时，才取对应连续子串调用 `detect_canonical_cnv`。
  3. **效果验证**：
     - `Ref = A–U–U–D–B`、`Alt = A–U–X–U–X–U–B`：`br_sh_rel = [0, 2, 4]`，`4 - 0 + 1 = 5 != 3`，不触发 `CNV`（仅报 `D` 的 `DEL` 与 `X` 相关 `INS`）；
     - `Ref = A–D–U–U–B`、`Alt = A–U–U–U–X–B`：`br_sh_rel = [0, 1, 2]`，`2 - 0 + 1 == 3`，正常检出 `CNV`（`k_ref=2, k_alt=3`）。

---

### 修改项 2（P1）：`INS` 与 `INV` 基因分型严格落实缺证据填 `.`（规格 §5.2）

- **目标文件**：`panflat/linearizer.py`（`_ContigAnalysis`、`_analyze_single_normal_fragment`、`_contig_covers_rank_interval` L3625–3654、`INV` 分型 L4183–4197、`INS` 分型 L4900–4964）

#### 2.1 `INS` 基因分型修复（L4900–4964）
- **根因定位**：
  - 规格 §5.2 规定 `INS` 取 `0` 必须同时满足：*“同一条 contig 的主链同时包含 `a` 和 `b`，并且两者之间与 `B(a,b)` 相同，或只有 `< min-sv-len` 的差异；其他情况填 `.`”*。
  - 现行代码在 `top_reps` 为空且未被 `repeat-guard` 屏蔽时，仅检查 `_contig_covers_anchor_pair(ca, rank_a, rank_b)` 就赋 `"0"`，导致在 `Ref = A–D(100bp)–B`、`Alt1 = A–X–B` 形成的组 `G(A, B)` 上，纯缺失 `100 bp D` 的 `Alt2 = A–B`（以及在 `(a, b)` 内发生 `≥ min_sv_len` 子区间缺失、串联扩增或远端参考块替换的其他样本）均被误填 `"0"`。
- **具体改法**：
  1. 在 Stage 6 构建每个 `contig_idx` 的终端偏离段索引（记录每个偏离段的 `(ta, tb, ref_span_bp, branch_bp)`，注意在 Stage 5 清空 `branch_provenance` 前保留或由 `branch_steps` 计算 `branch_bp`）。
  2. 对组 `G(rank_a, rank_b)` 中未产生非主干代表签名（`top_reps` 为空）且 `ckey not in guard_res.contig_had_masked_repeat` 的 contig `ca`：
     - 首先必须满足 `_contig_covers_anchor_pair(ca, rank_a, rank_b)`；
     - **情况 A（`c_list` 非空且全部归入 `rep0`，即 `r_idx == 0`）**：候选已通过 §3.6.3 相似聚类并入主干 `B(rank_a, rank_b)`；只要该偏离段在 `B(rank_a, rank_b)` 上不存在额外 `≥ min_sv_len` 的未覆盖参考缺失（且 `|branch_bp - ref_span_bp| < min_sv_len` 或无侧翼 `DEL`），判为 `"0"`，否则判为 `"."`；
     - **情况 B（`c_list` 为空）**：统计 `(rank_a, rank_b)` 开区间内**未进入 `ca.covered_ranks` 的参考 step 总碱基数 `uncovered_ref_bp`**，以及 `ca` 落在 `[rank_a, rank_b]` 内的所有终端偏离段的**分支侧总碱基数 `dev_branch_bp`**：
       - 仅当 `uncovered_ref_bp < config.min_sv_len` 且 `dev_branch_bp < config.min_sv_len`（例如完全一致，或仅有 `< 50 bp` 的 SNP / 微小 InDel）时，判为 `"0"`；
       - 一旦 `uncovered_ref_bp >= config.min_sv_len`（如 `A–B` 删除了 100 bp `D`，或子区间 `(a, m)` 删除了 100 bp `D1`）或 `dev_branch_bp >= config.min_sv_len`（如插入了 ≥50 bp 远端参考块或发生了 ≥50 bp 串联拷贝扩增），一律判为 `"."`。

#### 2.2 `INV` / `INV_DISPLACED` / `INV_EDGE` / `INV_CONFLICT` 基因分型修复（L3625–3654 & L4183–4197）
- **根因定位**：
  - `_contig_covers_rank_interval` 在 `left_flank` / `right_flank` 检查未通过时没有返回 `False`，而是回退执行了 `min_cov <= rmin and max_cov >= rmax`；
  - 导致任何仅覆盖倒位内部节点 `[rmin, rmax]` 而缺失两侧侧翼锚点的短 contig（如 `Ref = A–B–C` 下只含 `B⁺` 的 contig，甚至原本是 `B⁻` 却被 §3.3.0 整条 contig 定向翻转为 `B⁺` 的倒位内部碎片）都被误判为 `"0"`。
- **具体改法**：
  1. 在 `_ContigAnalysis` 中补充轻量字段 `normalized_inv_rank_spans: list[tuple[int, int]]`（记录该 contig 在 Stage 4 被归一化的 `INV` / `INV_DISPLACED` 原始秩区间 `(ch.rmin, ch.rmax)`，IPC 仅几个整数元组，零内存开销）。
  2. 重写 `_contig_covers_rank_interval(analysis, rmin, rmax, n_ref)`：
     - 若 `not analysis.has_main_chain`：返回 `False`；
     - 若 `analysis.unnormalized_rank_spans`、`analysis.large_jump_rank_spans` 或 `analysis.normalized_inv_rank_spans` 中存在与 `[rmin, rmax]` 相交的区间（对 `large_jump` 检查跨越断点或区间重叠）：返回 `False`（避免携带另一条部分重叠倒位的 contig 在归一化后被误算为当前倒位的正向参考型 `"0"`）；
     - **强制双侧外翼锚点覆盖**：
       - **左侧要求**：当 `rmin > 0` 时，正向主链必须覆盖严格位于倒位左侧的参考锚点（`min_cov < rmin`，若传入有效 `left_flank < rmin` 且覆盖则自然满足）；仅当倒位一直延伸到参考染色体最左端（`rmin == 0` 的左端 `INV_EDGE`）时，要求 `0 in analysis.covered_ranks` 且右侧有外翼锚点；
       - **右侧要求**：当 `rmax < n_ref - 1` 时，正向主链必须覆盖严格位于倒位右侧的参考锚点（`max_cov > rmax`）；仅当倒位延伸到参考染色体最右端（`rmax == n_ref - 1` 的右端 `INV_EDGE`）时，要求 `(n_ref - 1) in analysis.covered_ranks` 且左侧有外翼锚点；
       - 若 `rmin == 0 and rmax == n_ref - 1`（极端全染色体边界）：无外侧锚点可供独立定相，返回 `False`（填 `"."`）。

---

### 修改项 3（P1）：相似聚类补齐短序列判据 1 保护，并守住判据 3/4 边界

- **目标文件**：`panflat/repeat_guard.py`（`check_similarity_merge`，L836–976）

#### 3.1 关于前半句 `200=30I200=` 的设计边界说明（禁止误改点）
- **代数验证**：设 `min_sv_len = 50`、`cluster_win_id = 0.70`。对夹在完全匹配序列中间的长度为 $L$ 的纯小插入（`200= + L·I + 200=`），跨越该插入的 50 bp 滑动窗口内最多只有 $50 - L$ 个匹配列。若将纯插入列 `win_i` 直接计入分母 `win_aligned_cols = 50`：
  $$\frac{50 - L}{50} \ge 0.70 \iff L \le 15\text{ bp}$$
  这将导致 **16–49 bp** 的全部纯小插入（及 16–49 bp 末端悬垂）被判据 4 拒绝，使规格 §3.6.3 判据 3（`< min-sv-len` 即 `< 50 bp`）在 16–49 bp 范围完全失效，直接违背规格 **T5**（*“内部有分散的 SNP 或小 InDel（每个都 < 50 bp）→ 输出等于次级主干，不串接微小等位”*）与 **T13**（*“同时带 30 bp 新序列 → FASTA 不变”*）。
- **精细化加固（不伤及纯小插入）**：
  - 保留纯小插入（窗口内仅含 `=` 与 `I`、无错配/缺失补偿）不计入 `win_aligned_cols` 的现有行为，确保 `1–49 bp` 纯小插入（含 `200=30I200=`、`225=49I225=`）正常合并；
  - 同时收紧非对称替换块检测：若同一滑窗内插入与密集错配同时出现（例如 `win_i > 0` 且窗口内已比对列本身错配率高或伴随 `D`），现有公式 `win_m / (win_m + win_x + win_d_open)` 在 `15= + 5X + 30I` 这类“20 bp 替换为 50 bp 且含偶然匹配”的非对称替换块上可能得到 `15 / 20 = 0.75 >= 0.70`。可将 `is_compensating_indel_sub` 扩展为：`((win_i > 0) & (win_d_bp > 0)) | (win_d_open >= 4) | ((win_i > 0) & (win_x > 0) & (win_i + win_x >= int((1.0 - cluster_win_id) * win_len) + 1) & (win_x >= 4))`（即纯插入 `win_x == 0` 或仅夹 1–3 个零星 SNP 的 T5 场景完全不受影响，但含密集错配的非对称替换块按完整窗口惩罚拦截），并在函数文档字符串中明确注释判据 3 与判据 4 的职责边界。

#### 3.2 后半句修复：`≤ 1000 bp` `edlib` 分支补齐判据 1（多重命中、反向竞争与内部拆分比对拦截）
- **根因定位**：
  - 当 `len_s <= 1000 and len_r <= 1000` 时，代码直接调用 `edlib.align(s, r, mode="NW", task="path")`，未做任何多重命中或拆分比对检查：
    1. 当 `S` 为 200 bp 序列 `U`、`R` 为 400 bp 串联序列 `U + U` 时，`NW` 产生 `200=200D`（`gap_open_D = 1`，`ID_gc = 200/201 = 0.995`），无视 `S` 在 `R` 中存在**两处完全匹配**，违反判据 1；
    2. 当 `R` 含倒位重复 `U + revcomp(U)`，或 `S` 在 `R` 内部跨过一段 `≥ min_sv_len` 的大缺失（对应 `mappy` 下的拆分比对）时，`edlib` 同样会强行拉出一条全局路径放行。
- **具体改法**：
  1. **改用查询半全局内嵌比对 `mode="HW"`**：
     - 调用 `edlib.align(s, r, mode="HW", task="path")`（允许目标 `R` 两端自由悬垂而不产生伪末端 `D` gap，同时查询 `S` 的两端悬垂仍体现为 `I`/错配并受判据 3 约束，与 `> 1000 bp` 的 `mappy` 行为严格对齐）。
     - 解析最优比对在目标 `R` 上的覆盖区间 `[t_start, t_end)`（来自 `res["locations"][0]`）。
  2. **拦截最优多起点命中（Exact / Tied Multi-Hit）**：
     - 检查 `res["locations"]`：过滤掉同一命中末端微小 indel 等价偏移后，若存在两个最优区间 `[s1, e1]` 与 `[s2, e2]` 的重叠比例低于 `0.5`（例如 `[0, 199]` 与 `[200, 399]` 不重叠或错位超过 `min_sv_len`），说明 `S` 在 `R` 内有多个等价命中，立即返回 `False`。
  3. **拦截非重叠次优同向/反向多重命中（Near-Optimal Secondary / Reverse Hit）**：
     - 设次优同源命中编辑距离上限 `sec_max_ed = int(0.20 * len_s)`（对应 `mappy` `asm20` 约 80% 相似度的多命中检出能力）：
       - 在主命中区间外侧的左前缀 `r[:t_start]`（当 `t_start >= int(0.8 * len_s)`）与右后缀 `r[t_end:]`（当 `len_r - t_end >= int(0.8 * len_s)`）上运行 `edlib.align(s, sub_r, mode="HW", task="distance", k=sec_max_ed)`；若任一侧返回 `0 <= editDistance <= sec_max_ed`，判定为多重同向命中，返回 `False`；
       - 对反向互补序列 `rc_s = revcomp(s)` 在 `r` 上运行 `edlib.align(rc_s, r, mode="HW", task="locations", k=sec_max_ed)`；若存在与正向主命中不重合（或正反同分竞争）的可信反向命中，按判据 1 返回 `False`。
  4. **拦截内部大跨度缺失拆分比对（Split Alignment）**：
     - 在 `mode="HW"` 下，`R` 两端的正常悬垂已不进入 `ops`；此时 `ops` 中的 `'D'` 纯粹是 `S` 命中区间**内部**跳过的 `R` 碱基。若内部出现单段连续缺失 `length >= min_sv_len`（或内部累计 `del_bp >= min_sv_len`，同时保留 `< min_sv_len` 如 30 bp / `25+10=35 bp` 的小缺失正常合并），说明 `S` 在 `R` 上属于跨 SV 级缺口的拆分比对，返回 `False`（与 `mappy` 分支同样补充检查单段内部 `'D'` `< min_sv_len` 保持一致）。

---

### 修改项 4（P2）：精确判定首块 `join_before`，并在 `verify` 中闭环校验人工接头

- **目标文件**：
  - `panflat/models.py`（`CandidateBlock` / `ExtractedCandidate` 增加轻量布尔属性 `left_anchor_contiguous: bool = True`）
  - `panflat/linearizer.py`（`extract_candidate_blocks`、`screen_and_fold_candidate`、`steps_to_pieces` L1269–1318、`_solve_multi_member_component` L1818–1898、`assemble_pan_and_offsets` L2420–2458）
  - `panflat/verify_qc.py`（`verify_build_outputs` 第 2 节 Invariant II，L2376–2460）

#### 4.1 构建端（`linearizer.py`）修复
1. **候选提取与重复屏蔽阶段追踪左边界源连续性**：
   - 在 `extract_candidate_blocks` 中：仅当候选块起始于 `dev.branch_steps` 的下标 `0`（即中间没有跳过远端参考节点 `D` 或多拷贝分隔节点），且首个 step 未被左端截断（正向 `slice_start == 0`），且 `dev.branch_provenance[0]` 与紧邻左锚点 step 在同一源路径上同向紧邻时，设 `left_anchor_contiguous = True`；否则（如 `A–D–X–B–D` 剥离了 `D`，或偏离段起始处本身存在源不连续）设为 `False`；
   - 在 `screen_and_fold_candidate`（通用 `repeat-guard`）中：若未屏蔽区间 `u_start > 0`（即候选左端前缀被 `REPEAT_LIKE` 屏蔽切掉），生成的残余候选 `left_anchor_contiguous` 强制设为 `False`；若 `u_start == 0` 则继承原 `raw_cand.left_anchor_contiguous`（注意环折叠 `fold_tandem_cycles` 始终保留从下标 `0` 起的前 `max_copy >= 1` 份拷贝，不改变下标 `0` 的左接头连续性）。
2. **`steps_to_pieces` 与递归组件拼装传递真实首块接头类型**：
   - `steps_to_pieces(..., first_join_before=...)`：将 `len(pieces) == 0` 时的硬编码 `"source_contiguous"` 改为传入的 `first_join_before`（由 `exemplar.left_anchor_contiguous` 决定：连续为 `"source_contiguous"`，否则为 `"synthetic"`）；
   - `_solve_multi_member_component`：
     - 子问题在 `a_r == "START"`（`idx == 0`）提取子候选时，子候选若起始于分支下标 `0`，其左连续性还须与父成员 `d.exemplar.left_anchor_contiguous` 取逻辑与（`and`）；
     - 拼装次级主干起始段（`idx == 0` 且无前置 `START` 插入组）时，`cur_b_first_join` 初始化为 `b_sec.pieces[0].join_before` 而非硬编码 `"source_contiguous"`；若有前置 `START` 纯插入组且 `len(raw_out_pieces) == 0`，其首块同样保留 `rep.pieces[0].join_before`。
3. **顶层挂载最终复核（`assemble_pan_and_offsets`）**：
   - 在 `_emit_slot_blocks_at_step` 中，当 `p_i == 0` 且无前导 `spacer`（即 `not need_leading_spacer and rep_i == 0`）时：
     - 若紧邻在左锚点参考 step `r_prev = r_idx - 1` 之后（`r_idx > 0`），进一步核对首块 `(src_path, src_start, src_end, src_strand)` 在源路径上紧邻的前驱区间是否确实对应参考 `ref_steps[r_idx - 1]`（若 `r_idx == 0` 即染色体最左端前挂载，则要求为源路径起点，否则标 `"synthetic"`）。双重兜底确保任何路径下绝不漏标人工接头。

#### 4.2 独立校验端（`verify_qc.py`）修复
在 `verify_build_outputs` 的 **Global Invariant II** 检查中，除现有的 `"spacer"` `'N'` 串检查外，新增对 `"source_contiguous"` 与 `"synthetic"` 的严格几何与源路径核验：
1. **构建 GFA 源路径 step 边界索引**（复用 `verify_qc.py` 已解析的原始 GFA 路径信息，支持普通小图与 `> 10 MB` 流式大图）：
   - 记录参考路径每个 `ref_step` 的 `(node_id, orient, ref_len)`；
   - 对 `inserts.tsv` 中出现过 `"source_contiguous"` 的每条 `src_path`，索引其原始路径上的 step 坐标区间与 `(node_id, orient)`。
2. **逐行校验 `join_before`**：
   - **当 `join_before == "source_contiguous"` 时**：
     - **情况 I（`pan.fa` 中紧接在前一个插入 piece 之后，即 `prev_p_e == p_s`）**：必须满足 `prev["src_path"] == row["src_path"]` 且 `prev["src_strand"] == row["src_strand"]`，并且：
       - 若 `src_strand == "+"`：`prev_src_end == src_start`；
       - 若 `src_strand == "-"`：`prev_src_start == src_end`；
       - **凡不满足（例如把环折叠跳过 100 bp 的 `synthetic` 接头手动改成 `source_contiguous`，或跨来源无 spacer 拼接）立即报 `INV2` 错误退出！**
     - **情况 II（`pan.fa` 中直接紧接在参考 step `r_prev` 之后，即 `ref_step_pan_end == p_s`）**：
       - 设 `r_prev` 为 `(ref_nid, ref_orient, L_prev)`；
       - 若 `src_strand == "+"`：要求 `src_start >= L_prev`，且 `row["src_path"]` 上 `[src_start - L_prev, src_start)` 恰好为一个完整 step `(ref_nid, ref_orient)`；
       - 若 `src_strand == "-"`：要求 `src_end + L_prev <= len(src_path)`，且 `row["src_path"]` 上 `[src_end, src_end + L_prev)` 恰好为一个完整 step `(ref_nid, flip(ref_orient))`；
       - 若 `p_s == 0`（染色体首端无前驱）：仅当该 piece 自身位于源路径对应链的物理起点（`+` 链 `src_start == base_offset`，`-` 链 `src_end == base_offset + path_len`）才允许，否则报 `INV2` 错误；
     - **情况 III（`p_s` 前方存在未覆盖的 `'N'` spacer 空隙）**：无 spacer 才能标 `source_contiguous`，否则直接报 `INV2` 错误。
   - **当 `join_before == "synthetic"` 时**：
     - 同样校验 `p_s` 前方不存在未声明的 `'N'` 空隙（`p_s == 0` 或 `covered_arr[p_s - 1] in (1, 2)`）。

---

### 修改项 5（P2）：`reanchor_deviation` 改为显式栈迭代，彻底消除 `RecursionError`

- **目标文件**：`panflat/linearizer.py`（`reanchor_deviation`，L318–493）及 `panflat/gff_liftover.py`（`_visit_container`，L621–634）
- **根因定位**：
  - `reanchor_deviation` 通过区间严格收缩 `hi_ab - lo_ab < eff_hi - lo` 保证有限步终止（规格 §3.5 明确写明“不设 max-depth 参数”），但代码采用 Python 递归调用，嵌套层数达到 ~1,000 层时触发 Python 默认调用栈上限 `RecursionError`；
  - 同时原代码在每层执行 `next_ancestors = ancestor_keys | {parent_key}`（L448），深度为 $D$ 时产生 $O(D^2)$ 的 `frozenset` 拷贝开销。
- **具体改法**：
  1. **将 `reanchor_deviation` 重写为显式 LIFO 栈迭代（Iterative DFS）**：
     - 栈元素分两类：
       - 工作帧 `("WORK", cur_dev, cur_lo, cur_hi, cur_parent_key)`；
       - 回溯弹出帧 `("POP_ANC", added_parent_key)`；
     - 用单个可变集合 `active_ancestors: set[SubproblemStateKey] = set(ancestor_keys)` 原地维护当前 DFS 路径上的祖先键：进入子层前推入 `("POP_ANC", cur_parent_key)` 并 `active_ancestors.add(cur_parent_key)`，子层全部处理完弹栈时 `active_ancestors.discard(added_parent_key)`，时间复杂度从 $O(D^2)$ 降至 **$O(D)$**，空间复杂度 $O(D)$；
     - 对当前帧生成的 `raw_sub_devs` 先正向构造 `mapped_devs` 列表，再按 **`reversed(mapped_devs)` 逆序压入 LIFO 栈**，从而使出栈访问顺序与原递归左到右 DFS 顺序 **100% 相同**，`covered_ranks_out` 更新顺序与最终返回的 `refined` 列表与原递归实现**逐字节完全一致**。
  2. **同步加固其他潜在深层递归点**：
     - 将 `panflat/gff_liftover.py` 中的 `_visit_container`（GFF3 多层父子容器后序遍历）同步改为显式栈后序遍历；
     - 对极端合成图上进展条件 `(b)`（`solve_parallel_group` ↔ `_solve_multi_member_component`，深度受不同去重候选数 `|M|` 限制）补充按活跃候选规模自适应的安全栈余量保护，确保任意合法输入下均不抛 `RecursionError`。

---

## 4. 测试与验收方案

代码修改完成后，必须通过以下全部新增测试与既有回归门禁方可合入：

### 4.1 针对 5 项审查反例的专项回归测试（单元测试 + 黑盒 CLI E2E）
1. **反例 1 回归（分散节点不误报 `CNV` + 侧翼共存保留）**：
   - 构造 `Ref = A–U–U–D–B`、`Alt = A–U–X–U–X–U–B`，断言输出不含 `CNV`（仅含 `D` 的 `DEL` 与 `X` 的 `INS`）；同时验证 `Alt2 = A–U–U–U–B` 仍正常输出 `CNV (k_ref=2, k_alt=3)` + `DEL`。
2. **反例 2 回归（`INS` 与 `INV` 缺证据填 `.`）**：
   - `INS`：构造 `Ref = A–D(100bp)–B`、`Alt1 = A–X(100bp)–B`、`Alt2 = A–B`、`Alt3 = A–D_snp(100bp)–B`、`Alt4 = A–D–B`，断言 `INS` 行基因型中 `Alt2 == "."`、`Alt3 == "0"`、`Alt4 == "0"`，且 `n_called` 不计入 `Alt2`；
   - `INV`：构造 `Ref = A–B–C`、`Alt1 = A–B⁻–C`、短 contig `Alt2 = B⁺`、短反向 contig `Alt3 = B⁻`（经 §3.3.0 定向翻为 `B⁺`）、完整正向 `Alt4 = A–B–C`，断言 `INV` 行中 `Alt2 == "."`、`Alt3 == "."`、`Alt4 == "0"`。
3. **反例 3 回归（短序列双命中拒绝合并 + `< 50 bp` 纯小插入保持合并）**：
   - 构造 200 bp 候选 `U` 与 400 bp 代表 `U + U`（及 `U + U_2snp`、`U + revcomp(U)`），断言 `check_similarity_merge` 返回 `False`，端到端构建保留为两个独立等位；
   - 同时断言 `200=30I200=` 与 `225=49I225=` 仍返回 `True`（守护 **T5** 与判据 3 边界 `49 bp` vs `50 bp`），而 `50 bp` 插入及非对称替换块返回 `False`。
4. **反例 4 回归（`source_contiguous` 精确标注 + `verify` 篡改拦截）**：
   - 构造 `Ref = A–B–D`、`Alt = A–D–X–B–D`（以及 `repeat-guard` 切掉左端前缀的纯插入），断言提取出的 `X` 首块在 `inserts.tsv` 中 `join_before == "synthetic"`；
   - 构造环折叠 `U–U–Tail` 产生 2 个 piece（`source_contiguous` + `synthetic`），将第 2 个 piece 的 `join_before` 手动篡改为 `"source_contiguous"`（以及在 `A–D–X–B–D` 输出上把首块篡改为 `"source_contiguous"`），独立运行 `panflat verify` 断言返回非零退出码并报 `INV2` 错误。
5. **反例 5 回归（`1,050+` 层深层嵌套局部重锚）**：
   - 构造 `1,200` 层嵌套重复区间包裹末端 `60 bp` 新插入 `X` 的合法合成 GFA，断言 `panflat build` 与 `panflat verify` 零异常通过、准确回收 `60 bp` 插入且不触发 `RecursionError`。

### 4.2 全量门禁检查
- `ruff check --select F,E9,W panflat tests`：`0` 错误；
- `! rg -n '^\s*(import|from)\s+panflat\b' tests/e2e`：E2E 测试 `0` 内部白盒导入；
- `git diff --exit-code panflat_v1_final_revised.md`：规格文档 `0` 字节改动；
- 全量既有测试套件（`543+` 用例）+ 新增专项回归测试 `100%` 通过，且 `--threads 1` 与 `--threads 4` 输出 SHA-256 逐字节一致（T22）。
