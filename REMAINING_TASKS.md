# PanFlat v1 — 剩余编码与验证任务完成报告（`REMAINING_TASKS.md`）

- **最终状态**：**`ALL COMPLETED`（P1 全部 4 项编码与回归测试、P2 文档与测试计数同步、P3 全量门禁复审与 30 单倍型染色体规模冒烟验证已 100% 完成）**
- **全量验证结果**：
  - 全量测试套件：**`523 / 523` 全部通过**（`170` 个单元测试 + `353` 个黑盒 CLI E2E 测试，总耗时 `106.27s`）
  - 静态检查：`ruff check --select F,E9,W panflat tests` **`0` 错误**
  - E2E 黑盒纯度：`! rg -n '^\s*(import|from)\s+panflat\b' tests/e2e` **`0` 内部导入**
  - 规范零改动：`git diff --exit-code panflat_v1_final_revised.md` **`0` 字节改动**

---

## 一、前期已完成的 M7 修复基线（M7 Round 1 + Round 2）

1. **`W` 行 `LazyStepTuple` 与四类图编码完全等价**（`panflat/gfa_io.py`、`panflat/normalizer.py`、`panflat/linearizer.py`、`panflat/centromere.py`）：修复了 `W` 行路径在紧凑模式下漏标 `multi_hit`、漏算 `DEL`/`CNV`、倒位与着丝粒锚点定位不一致，以及 `P`/`W` 精确同名 `src_path` 冲突未拦截的问题。
2. **PanSN 与非 PanSN 参考路径消歧及 `--backbone-prefix` 覆盖校验**（`panflat/gfa_io.py`、`panflat/verify_qc.py`）：强制 `#` 分隔符边界匹配，支持 `W` 行 `:start-end` 坐标后缀剥离、非 PanSN 精确匹配优先消歧，以及 `panflat merge` 后对非 PanSN / `W` 行路径显式 `--backbone-prefix` 覆盖校验。
3. **紧凑节点存储（`CompactNodeStore` / `CompactRefNodeSet`）安全边界**（`panflat/models.py`、`panflat/gfa_io.py`）：强制规范十进制正整数双射（拒绝前导零如 `01` 与 `> 50,000,000` 稀疏节点 ID 走紧凑数组），并补齐 `_extra` 回退与负编码绝对值边界检查。
4. **同位点 `CNV` + `DEL` 并存输出与多等位部分覆盖 `CNV` 分型**（`panflat/linearizer.py`）：移除了 `CNV` 对同位点 `DEL` 的静默抑制；修复了多等位 `CNV` 下单侧截断 contig 误赋具体等位号而非 `.` 的问题。
5. **两阶段 `--qc` 自比对屏蔽与连续参考比对**（`panflat/verify_qc.py`）：Stage 1 改为比对完整连续参考染色体序列（消除节点边界切碎导致的漏检），Stage 2 仅排除同一 `pan.fa` 物理区间自命中并将单 piece 目标裁剪至自身边界（消除跨等位 `N` spacer 嵌合比对）。
6. **多染色体结合律 `panflat merge` 与大文件校验加固**（`panflat/writers.py`、`panflat/verify_qc.py`）：支持扁平/左嵌套/右嵌套任意层级合并，`unresolved` 记录强制 `{chrom}_UNRES_` 前缀隔离，同步重映射 `ref.lifted.gff3` 中的 `sv_id`；补齐 `> 2 MB` `offsets.tsv` 快速路径的节点序列逐碱基校验、多链 `.chain` 块注释按链绑定，以及畸形 FASTA/GFA 严格拒绝。
7. **多父级（Multi-`Parent`）GFF3 平移与同源异构体独立校验**（`panflat/gff_liftover.py`、`panflat/verify_qc.py`）：支持逗号分隔多 `Parent` 共享 `exon`/`CDS` 的按转录本独立拆分与蛋白还原校验。

---

## 二、P1 编码与回归测试任务（全部已完成 ✅）

### 任务 1.1：修复多可变剪接转录本下 `gene` 容器特征的 `sv_status` 叠加聚合缺失 — ✅ 已完成
- **修改文件**：`panflat/gff_liftover.py`（`liftover_gff3`）
- **实现摘要**：
  - 采用自底向上的 DFS 后序遍历（`container_postorder`）处理非可拆分容器特征：
    - 叶级转录本容器（`mRNA` / `transcript` / `lnc_RNA`）严格仅按自身直属 `desc_cds` / `desc_utrs` / `desc_exons` 评估插入点，保证兄弟转录本共享 `exon` 时不串染；
    - 父级容器（`gene` 等）并集所有直接子容器的 `sv_statuses` 与 `sv_ids`（同时对不隶属任何子容器的直接子特征或超出子容器范围的区间做补充评估），并按规范固定顺序 `("intronic_insertion", "utr_insertion", "cds_split_by_insertion")` 输出。
- **回归测试**：
  - 单元测试：`tests/unit/test_m7_deep_review_hardening.py::test_m7_30_p1_1_gff_gene_multi_isoform_sv_status_union_without_sibling_bleed`（PASS）
  - CLI E2E 测试：`tests/e2e/test_tier5_adversarial.py::test_tier5_m7_15_p1_1_and_p1_2_gff_gene_isoform_union_and_per_allele_ins_tags_cli`（PASS）

---

### 任务 1.2：隔离多等位 `INS` 事件 `alleles` 字段中的等位级标签（`partial_content=1` / `repeat_residual=1` / `multi_hit=1`）— ✅ 已完成
- **修改文件**：`panflat/linearizer.py`（`ComponentResult`、`ClusterRepresentative`、`ParallelGroupSolution`、`_solve_multi_member_component`、`solve_parallel_group`、`run_chromosome_pipeline`）
- **实现摘要**：
  - 新增 `cand_to_flags: dict[str, tuple[bool, bool, bool]]` 精确追踪每个候选块在局部重锚、次级主干组件递归（含局部不可拆解多拷贝节点 `repeat_residual`）及聚类代表合并后的等位级 `(partial_content, repeat_residual, multi_hit)` 标志，杜绝 `A+B` 与 `A+C` 组件分支间跨等位串染；
  - 移除了 `run_chromosome_pipeline` 中将整个平行分支组 `G(a,b)` 级别的 `group_has_partial` / `group_has_rep_res` / `group_has_mh` 无差别广播到所有等位的逻辑。
- **回归测试**：
  - 单元测试：`tests/unit/test_m7_deep_review_hardening.py::test_m7_31_p1_2_multiallelic_ins_per_allele_tag_isolation`（PASS）
  - CLI E2E 测试：`tests/e2e/test_tier5_adversarial.py::test_tier5_m7_15_p1_1_and_p1_2_gff_gene_isoform_union_and_per_allele_ins_tags_cli`（PASS）

---

### 任务 1.3：强化着丝粒 `CEN_ISLAND` 挂载点局部偏移差的精确等式校验（规范 §3.8.5）— ✅ 已完成
- **修改文件**：`panflat/verify_qc.py`（`verify_build_outputs` 第 5 节）、`tests/e2e/helpers.py`（`assert_centromere_accounting_and_offsets`）
- **实现摘要**：
  - 同时在常规路径与 `is_large_offsets`（`> 2 MB` `offsets.tsv`）快速路径、内部挂载点 `r_e < ref_total_len` 与染色体末端 `r_e == ref_total_len` 下精确求取槽位区间 `[slot_pan_start, slot_pan_end)` 与 `local_offset_delta = slot_pan_end - slot_pan_start`；
  - 核对槽位内所有 `CEN_ISLAND` piece 排在同位置普通 `INS` piece 之前、各 piece 严格递增不重叠、所有间隙全为 `'N'`、首尾及相邻事件间仅含单段 `eff_spacer_n`（严禁双倍叠加 `2 * eff_spacer_n`），并强制校验精确等式 `local_offset_delta == slot_ins_len == cen_bp + normal_ins_bp + expected_spacer_bp`（以及 `0` 代表区域在无普通 `INS` 共挂载时新增长度严格为 `0`）。
- **回归测试**：
  - 单元测试：`tests/unit/test_m7_deep_review_hardening.py::test_m7_32_p1_3_verify_centromere_mount_exact_local_offset_and_stacked_spacer_rejection`（PASS）
  - CLI E2E 测试：`tests/e2e/test_tier5_adversarial.py::test_tier5_m7_16_p1_3_and_p1_4_cen_mount_stacked_spacer_and_unresolved_header_tamper_cli`（PASS）

---

### 任务 1.4：对齐 `unresolved.fa` FASTA Header 的独立 `strand=<±>` 字段格式与校验 — ✅ 已完成
- **修改文件**：`panflat/writers.py`（`write_build_outputs`、`merge_chromosomes`）、`panflat/verify_qc.py`（`_parse_fasta_with_header_kv`、`verify_build_outputs` 第 4 节）、`tests/e2e/helpers.py`（`assert_all_invariants`）
- **实现摘要**：
  - 单染色体构建与多染色体 `merge_chromosomes` 统一输出规范 §5 格式：`>{id} reason={reason} src={src_path}:{src_start}-{src_end} strand={src_strand} region_id={region_id} event_id={event_id} len={length}`；
  - `verify_build_outputs` 解析并核对 `unresolved.fa` 每条记录的 `src`、`strand`、`reason`（兼容 §5.5 同源同向去重共享 `exported_fasta_id` 的多原因关联集合），并拦截未被 `unresolved.tsv` 引用的孤儿 FASTA 记录。
- **回归测试**：
  - 单元测试：`tests/unit/test_m7_deep_review_hardening.py::test_m7_33_p1_4_unresolved_fa_header_strand_and_verify_tamper_rejection`（PASS）
  - CLI E2E 测试：`tests/e2e/test_tier5_adversarial.py::test_tier5_m7_16_p1_3_and_p1_4_cen_mount_stacked_spacer_and_unresolved_header_tamper_cli`（PASS）

---

## 三、P2 文档与测试元数据同步任务（已完成 ✅）

### 任务 2.1：同步 `TEST_READY.md` 与 `PROJECT.md` 的测试统计数据 — ✅ 已完成
- **实测精确收集与通过数**：**`523 / 523`**（单元测试 **`170`** 个 + 黑盒 CLI E2E 测试 **`353`** 个）：
  - `tests/unit/test_m1_foundation.py`：`34`
  - `tests/unit/test_m2_normalizer_centromere.py`：`44`
  - `tests/unit/test_m3_linearizer.py`：`49`
  - `tests/unit/test_m4_liftover_writers_verify_cli.py`：`10`
  - `tests/unit/test_m7_deep_review_hardening.py`：`33`
  - `tests/e2e/test_tier1_features.py`：`133`
  - `tests/e2e/test_tier2_boundaries.py`：`132`
  - `tests/e2e/test_tier3_combinations.py`：`28`
  - `tests/e2e/test_tier4_scenarios.py`：`14`
  - `tests/e2e/test_tier5_adversarial.py`：`46`
- 已同步更新 `TEST_READY.md` 与 `PROJECT.md`。

---

## 四、P3 门禁复审闭环与染色体规模多单倍型冒烟复核（已完成 ✅）

### 任务 3.1：全量静态检查、黑盒纯度检查与回归门禁复审 — ✅ 已完成
- `ruff check --select F,E9,W panflat tests` -> `All checks passed!`（`exit 0`）
- `! rg -n '^\s*(import|from)\s+panflat\b' tests/e2e` -> `0` 匹配（`exit 0`）
- `git diff --exit-code panflat_v1_final_revised.md` -> `0` 差异（`exit 0`）
- `pytest -p no:cacheprovider -q` -> `523 passed in 106.27s`（`exit 0`）

### 任务 3.2：染色体规模 30 单倍型双染色体端到端冒烟复核 — ✅ 已完成
- 在云端环境合成并执行了覆盖全部大文件快速路径（`CompactNodeStore` + `CompactRefNodeSet` + `LazyStepTuple`、`> 10 MB` GFA 按需流式切片校验、`> 2 MB` `offsets.tsv` `is_large_offsets` 快速路径）的 **30 单倍型（`1` 参考 + `20` `P` 行单倍型 + `9` 断裂/反向 `W` 行单倍型）、双染色体（`Chr01` `4.50 Mb` / `45,000` 参考步 / `15.46 MB` GFA / `2.27 MB` `offsets.tsv`；`Chr02` `2.00 Mb` / `20,000` 参考步 / `7.05 MB` GFA；合并 `22.51 MB` GFA / `3.26 MB` `offsets.tsv`）** 全流程基准测试：
  1. `panflat build --qc --threads 8 --centromere-mode extract-islands --ref-gff`（`Chr01`）：**`2.64s`**，峰值内存 **`209.3 MiB`**（`0.204 GiB`）；
  2. `panflat build --qc --threads 8 --centromere-mode extract-islands --ref-gff`（`Chr02`）：**`1.51s`**，峰值内存 **`183.7 MiB`**（`0.179 GiB`）；
  3. 独立 `panflat verify`（`Chr01`，含 `--backbone-prefix Ref#1# --centromere-bed --ref-gff`）：**`0.50s`**，峰值内存 **`183.7 MiB`**；
  4. `panflat merge`（合并 `Chr01` + `Chr02`）：**`0.20s`**，峰值内存 **`183.7 MiB`**；
  5. 独立 `panflat verify`（合并后双染色体 `merged_pan`，`22.51 MB` GFA + `3.26 MB` `offsets.tsv`）：**`0.63s`**，峰值内存 **`183.7 MiB`**；
  6. 临时目录 `/tmp/panflat_chr_smoke` 已自动清理完毕，工作区保持纯净。
