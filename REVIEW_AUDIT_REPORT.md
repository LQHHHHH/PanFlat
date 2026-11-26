# `REVIEW_FIX_PLAN.md` 修复结果代码审查报告（`REVIEW_AUDIT_REPORT.md`）

> **审查对象**：`LQHHHHH/PanFlat` 主分支提交 `d369cd2`（`fix: implement all REVIEW_FIX_PLAN.md fixes and regression tests`）  
> **对照基准**：[`REVIEW_FIX_PLAN.md`](REVIEW_FIX_PLAN.md) 与设计规格 [`panflat_v1_final_revised.md`](panflat_v1_final_revised.md)  
> **总体结论**：**通过验收**。5 项外部审查问题全部按设计意图闭环修复，5 条设计红线与全量门禁（`585/585` 测试）100% 达标；白盒对抗探针另定位出 **1 项真实大泛基因组宽区间边界隐患（P2）** 与 **1 项模块拆分可维护性建议（P3）**，留存备查。

---

## 1. 验收门禁与对抗探针实测总览

| 检查维度 | 执行命令 / 测试规模 | 实测结果 |
|---|---|---|
| **静态代码质量** | `ruff check --select F,E9,W panflat tests` | **通过**（`0` 错误 / 告警） |
| **E2E 黑盒纯粹性** | `! rg -n '^\s*(import\|from)\s+panflat\b' tests/e2e` | **通过**（`0` 内部导入，纯 CLI 子进程执行） |
| **规格文档零改动红线** | `git diff --exit-code panflat_v1_final_revised.md` | **通过**（`0` 字节改动） |
| **全量单元 + E2E 回归套件** | `pytest -q`（含既有套件 + 新增专项回归） | **通过**（`585 passed in 297.91s`，含 `--threads 1` vs `--threads 4` SHA-256 逐字节一致） |
| **附加探针 A：`edlib` 随机模糊测试** | `<= 1000 bp` 随机生成 `1,000` 组（双拷贝同向、正反互补混合、`>= 50 bp` 拆分缺失、`>= 50 bp` 插入、`16–49 bp` 纯小插入） | **100% 通过**；900–980 bp 同聚物/回文/双拷贝最坏单次延迟仅 **`0.25–1.02 ms`** |
| **附加探针 B：超大图流式边界与篡改校验** | 合成 `100,050` 个参考 step、`12.86 MB` GFA、`5.13 MB` `offsets.tsv`（同时跨越 `n_ref > 100,000` 清空 `alt_fragments` 阈值与 `> 10 MB` GFA 流式解析阈值） | **100% 通过**；剥离远端节点、整条反向 `-` P-line、倒位归一化边界接头的 `join_before` 全部准确，篡改后 `panflat verify` 稳定退出码 `1` 并报 `[INV2]` |

---

## 2. 五项修复逐项白盒复核

### 2.1 修改项 1（P1）：`CNV` 分散节点假阳性与侧翼共存保留 — **完全正确**
- **核心代码**：
  - 紧凑路径（`LazyStepTuple`）：[`panflat/linearizer.py`](panflat/linearizer.py#L5176-L5215)
  - 普通路径（`Sequence[Step]`）：[`panflat/linearizer.py`](panflat/linearizer.py#L5286-L5334)
- **复核结论**：
  1. 在验证参考侧共享节点连续（`sh_rel[-1] - sh_rel[0] + 1 == n_sh`）的基础上，紧凑路径与普通路径均补齐了**分支侧共享步原始下标连续性约束**（`br_sh_rel[-1] - br_sh_rel[0] + 1 == n_br_sh`，普通路径同时强制 `_is_full_step(st)` 排除 `repeat-guard` 半截切片）。
  2. 反例 `Ref = A–U–U–D–B`、`Alt = A–U–X–U–X–U–B` 中 `br_sh_rel = [0, 2, 4]` 不连续，不再触发伪串联 `CNV`，仅输出 `DEL(D)` 与 `INS(X)`。
  3. 红线场景 `Ref = A–D–U–U–B`、`Alt = A–U–U–U–B` / `A–U–U–U–X–B` 中 `br_sh_rel = [0, 1, 2]` 连续，正常输出共存的 `DEL + CNV(k_ref=2, k_alt=3) + INS` 并共享同一 `region_id`。

### 2.2 修改项 2（P1）：`INS` 与 `INV` 缺证据填 `.`（规格 §5.2）— **反例修复正确（宽区间有一处 P2 边界隐患）**
- **核心代码**：
  - `INV` / `INV_DISPLACED` / `INV_EDGE` / `INV_CONFLICT` 覆盖判定：[`_contig_covers_rank_interval`](panflat/linearizer.py#L3996-L4033)、[`panflat/linearizer.py`](panflat/linearizer.py#L3214-L3265)
  - 锚点对覆盖判定：[`_contig_covers_anchor_pair`](panflat/linearizer.py#L3975-L3993)
  - `INS` 情况 A（候选并入 `rep0`）校验：[`_case_a_rep0_matches_backbone`](panflat/linearizer.py#L4788-L4884)
  - `INS` 情况 B（无保留候选）校验：[`_uncovered_ref_bp_open`](panflat/linearizer.py#L4738-L4765)、[`_dev_branch_bp_in_span`](panflat/linearizer.py#L4767-L4786)、[`panflat/linearizer.py`](panflat/linearizer.py#L5551-L5596)
- **复核结论**：
  1. **`INV` 分型**：彻底移除了旧版回退逻辑 `min_cov <= rmin and max_cov >= rmax`，强制双侧严格外翼锚点（`min_cov < rmin` 与 `max_cov > rmax`，并精确兼容左/右染色体端点的 `INV_EDGE`）；同时记录 `normalized_inv_rank_spans` 与 `unnormalized_gap_spans`，防止携带部分重叠倒位、单节点 `LARGE_JUMP` 或经 §3.3.0 整条翻转的内部碎片 `B⁻` 被误填为 `"0"`。
  2. **`INS` 分型（情况 A）**：在 [`solve_parallel_group`](panflat/linearizer.py#L2246-L2262) 中缓存 `rep0_cand_spans`，由 [`_case_a_rep0_matches_backbone`](panflat/linearizer.py#L4788-L4884) 核验主干覆盖度，连“删掉 `100 bp D1` 后剩余 `D2_snp` 在半全局 `HW` 下并入 `rep0 = D1+D2`”的隐蔽反例也能准确拦截为 `"."`。
  3. **`INS` 分型（情况 B）**：单偏离段下（如 `Ref = A–D(100bp)–B`、`Alt2 = A–B`）准确输出 `"."` 并从 `n_called` 中剔除；但在跨越多个内部锚点的大型区间 `G(a, b)` 上存在独立 SNP 跨段累加过严问题（见第 4.1 节）。

### 2.3 修改项 3（P1）：相似聚类补齐短序列判据 1 保护并守住判据 3/4 边界 — **完全正确**
- **核心代码**：
  - 统一入口与判据 2/3/4：[`_check_similarity_merge_with_span`](panflat/repeat_guard.py#L5814-L6094)、[`check_similarity_merge`](panflat/repeat_guard.py#L6097-L6141)
  - `<= 1000 bp` 仿射重比对、短臂恢复与判据 1 唯一性校验：[`_realign_edlib_edit_clusters`](panflat/repeat_guard.py#L923-L1015)、[`_trim_edlib_terminal_overhangs`](panflat/repeat_guard.py#L1064-L1197)、[`_recover_edlib_abandoned_arms`](panflat/repeat_guard.py#L1261-L2158)、[`_check_edlib_criterion1_unique`](panflat/repeat_guard.py#L5589-L5811)
- **复核结论**：
  1. **判据 3/4 边界红线（3a）**：保持纯小插入（如 `200=30I200=`、`225=49I225=`）由判据 3（`< min_sv_len`）管辖而不挤占 50 bp 滑窗分母，完整守住规格 **T5** 与 **T13**；同时对称收紧了插入/缺失伴随密集错配的非对称替换块惩罚（[`panflat/repeat_guard.py`](panflat/repeat_guard.py#L6070-L6089)）。
  2. **短序列判据 1 保护（3b）**：将 `edlib` 从全局 `NW` 切换为查询半全局 `HW`，配合局部 `asm20` 双段仿射重比对与悬垂修剪，完整拦截同向最优/次优多命中（`U` vs `U+U` / `U+U_2snp`）、反向互补竞争命中（`U` vs `U+revcomp(U)` 及回文同分竞争）以及内部单段或累计 `>= min_sv_len` 的大缺失拆分比对。

### 2.4 修改项 4（P2）：`join_before` 源连续性精确追踪与独立 `verify` 闭环校验 — **完全正确**
- **核心代码**：
  - 数据模型与偏离段边界：[`panflat/models.py`](panflat/models.py#L343-L370)、[`_step_boundary_contiguous`](panflat/normalizer.py#L40-L65)、[`panflat/normalizer.py`](panflat/normalizer.py#L1354-L1402)、[`panflat/linearizer.py`](panflat/linearizer.py#L3474-L3480)
  - 候选提取、重复屏蔽与组件拼装：[`extract_candidate_blocks`](panflat/linearizer.py#L676-L678) & [L859-L863](panflat/linearizer.py#L859-L863)、[`screen_and_fold_candidate`](panflat/linearizer.py#L1197-L1199)、[`steps_to_pieces`](panflat/linearizer.py#L1349-L1401)、[`_solve_multi_member_component`](panflat/linearizer.py#L1713-L1720) & [L1920-L1994](panflat/linearizer.py#L1920-L1994)、[`assemble_pan_and_offsets`](panflat/linearizer.py#L2549-L2697)
  - 独立校验端：[`_parse_gfa_paths_and_walks`](panflat/verify_qc.py#L790-L1368)、[`verify_build_outputs`](panflat/verify_qc.py#L2752-L2992)
- **复核结论**：
  1. 构建端完整覆盖了远端参考节点剥离（`A–D–X–B–D`）、`repeat-guard` 左前缀切除（`u_start > 0`）、倒位归一化断点边界（`LazyProvenanceTuple._inv_intervals`）、环折叠内部断点与多成员递归组件 `"START"` 传递，同时保留了正向、整条翻转 `-` contig（**T23**）和倒位内部纯插入（**T7**）首块的合法 `"source_contiguous"`。
  2. 独立校验端 [`verify_build_outputs`](panflat/verify_qc.py#L2897-L2992) 对所有 `"source_contiguous"` 行闭环验证了前驱紧邻 piece（情况 I）、前驱参考 step 在源路径对应链上的精确相邻节点与方向（情况 II）以及无未声明 `'N'` 间隔（情况 III），彻底封堵了人工篡改绕过 `verify` 的漏洞。

### 2.5 修改项 5（P2）：深层嵌套局部重锚迭代化消除 `RecursionError` — **完全正确**
- **核心代码**：
  - 局部重锚显式栈：[`reanchor_deviation`](panflat/linearizer.py#L327-L544)
  - GFF3 容器后序显式栈：[`liftover_gff3_lines`](panflat/gff_liftover.py#L618-L646)
  - 组件递归自适应栈余量保护：[`_solve_multi_member_component`](panflat/linearizer.py#L1746-L1780)
- **复核结论**：
  1. [`reanchor_deviation`](panflat/linearizer.py#L345-L543) 采用 `("WORK", ...)` + `("POP_ANC", ...)` 双帧 LIFO 栈原地维护单个可变集合 `active_ancestors`，并对子偏离段执行 `reversed(mapped_devs)` 压栈，遍历顺序与输出同原递归实现 **100% 逐字节一致**，祖先维护开销由 $O(D^2)$ 降至 $O(D)$。
  2. GFF3 容器树/DAG 后序遍历（及已有的 BFS `_get_descendants`）无任何 Python 递归调用，`1,200+` 层嵌套测试稳定通过。

---

## 3. 既有测试文件改动合理性核查

本次提交除新增 [`tests/unit/test_review_fix_regressions.py`](tests/unit/test_review_fix_regressions.py)（3,859 行）与 [`tests/e2e/test_tier5_review_fixes.py`](tests/e2e/test_tier5_review_fixes.py)（1,562 行）外，仅微调了 2 处既有测试，经核查均完全正当：

1. **`C20` 重复屏蔽双残余首块接头修正**（[`tests/e2e/test_tier1_features.py`](tests/e2e/test_tier1_features.py#L2331-L2337)、[`tests/unit/test_m3_linearizer.py`](tests/unit/test_m3_linearizer.py#L554-L559)）：该用例中右侧残余 `[2000, 2600)`（600 bp，源区间 `2250-2850`，`u_start = 2000 > 0`）因长于左侧残余 `[0, 500)`（500 bp）按规格长度降序排作第一代表，紧接在 `n1`（源区间 `0-250`）之后跳过了 `2,000 bp`。旧断言 `"source_contiguous"` 恰系修改项 4 的历史漏洞表现，更新为 `"synthetic"` 完全符合全局不变量 II。
2. **`1000 bp / 1001 bp` 边界用例首碱基碰撞修正**（[`tests/unit/test_m3_linearizer.py`](tests/unit/test_m3_linearizer.py#L1627-L1630)）：`seed=4415` 与 `seed=4416` 首碱基恰好同为 `'T'`，导致预期 `50 bp` 的末端分叉实际只有 `49 bp`（`< min_sv_len`）；消除首碱基碰撞后准确测试了 `50 bp` 悬垂边界。

---

## 4. 留存备查问题与后续优化建议（非当前阻塞项）

> [!NOTE]
> 以下两项不阻塞当前 `d369cd2` 合入验收，可在后续迭代（如推进 [`DEDUP_PLAN.md`](DEDUP_PLAN.md) 时）视需要一并优化。

### 4.1 [P2 边界隐患] 跨多锚点宽区间 `G(a, b)` 的 `INS` 基因分型（情况 B）对分散微小差异做了跨区间累加

- **代码位置**：[`panflat/linearizer.py`](panflat/linearizer.py#L5569-L5595) 调用的 [`_uncovered_ref_bp_open`](panflat/linearizer.py#L4738-L4765) 与 [`_dev_branch_bp_in_span`](panflat/linearizer.py#L4767-L4786)。
- **现象与根因**：
  代码严格实现了 [`REVIEW_FIX_PLAN.md`](REVIEW_FIX_PLAN.md#L71-L73) 中情况 B 的区间求和公式（要求整个 `(rank_a, rank_b)` 内未覆盖参考 step 总碱基数 `< min_sv_len` 且分支偏离总碱基数 `< min_sv_len`）。在单偏离段下完全准确，但当某个替换型 `INS` 的 `(rank_a, rank_b)` 横跨大量参考节点（例如梨 `Chr06A` 上跨几千至上万个 step 的宽区间 `INS`）时，共线参考型样本 `Alt2` 在区间内部往往被局部锚点切分为许多独立的微小子偏离段：
  1. **零星 SNP 节点累加误判**：例如 `Ref = A – N1(30bp) – U(100bp) – N2(30bp) – B`，`Alt1 = A – X(200bp) – B` 产生 `INS(A..B)`，而 `Alt2` 在 `N1` 与 `N2` 上各仅有 **1 个 SNP**（无任何缺失/插入，总差异仅 2 bp）。`Alt2` 在同区间的 `DEL(A..B)` 行被正确判为 `"0"`，但在 `INS(A..B)` 行因 `30 + 30 = 60 >= 50` 被判为 `"."`；同理，10 kb 宽区间内只要存在 50 个 `1 bp` SNP 气泡（`99.5%` 一致度），共线参考型单倍型都会在 `INS` 行退化为 `"."`，压低宽区间 `INS` 的 `n_called`。
  2. **子组已并入 `rep0` 的 SNP 节点未豁免**：例如 `Ref = A – M1(60bp) – U(100bp) – B`，`Alt2` 在 `M1(60bp)` 上仅有 1 个 SNP；尽管在子组 `G(A, U)` 中 `M1_snp` 已通过相似聚类并入 `rep0`，外层 `G(A, B)` 仍把 `M1`（60 bp）计入未覆盖/偏离总长而判为 `"."`。
- **后续修法备忘**：
  由于同时覆盖 `rank_a` 与 `rank_b` 的 contig 在区间内部除自身终端子偏离段 `(ta_i, tb_i)` 外均与参考 100% 一致，后续只需改为**逐个子偏离段判定**：豁免在各自子组中已通过 [`_case_a_rep0_matches_backbone`](panflat/linearizer.py#L4788-L4884) 并入 `rep0` 的子偏离段，仅当 `[rank_a, rank_b]` 内存在任一未并入 `rep0` 且自身 `ref_span_bp >= min_sv_len` 或 `br_bp >= min_sv_len`（或含重复屏蔽残余）的子偏离段时才判 `"."`，其余判 `"0"`。

### 4.2 [P3 可维护性建议] 拆分 `panflat/repeat_guard.py` 中的短序列仿射比对实现

- **代码位置**：[`panflat/repeat_guard.py`](panflat/repeat_guard.py)（当前总长 **6,142 行**，较修改前增加 5,363 行，其中 [`_has_distinct_edlib_indel_copy2`](panflat/repeat_guard.py#L2816-L4378) 为 1,563 行、[`_has_distinct_edlib_del_split_copy`](panflat/repeat_guard.py#L4381-L5586) 为 1,206 行）。
- **评估与建议**：
  这部分代码用于将 `<= 1 kb` 的单位编辑距离 `edlib mode="HW"` 精确对齐到 `mappy asm20` 双段仿射罚分语义，并通过零假阴性 C 层预筛把最坏耗时控制在 `~1 ms` 内，功能与性能实测均可靠。在下一步开展 [`DEDUP_PLAN.md`](DEDUP_PLAN.md) 改造前，建议将 `edlib` 仿射重比对与判据 1 多命中检测独立拆分为单独模块（如 `panflat/edlib_similarity.py`），使 [`panflat/repeat_guard.py`](panflat/repeat_guard.py) 聚焦于区间运算、15-mer 重复掩码、`mappy` 区间减法与顶层聚类入口。
