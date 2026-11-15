# PanFlat v1 — 剩余编码与验证任务清单（`REMAINING_TASKS.md`）

- **盘点基线**：M7 Round 1 + Round 2 第一轮修复已完成，当前工作区全量测试 **`517 / 517` 全部通过**（`166` 个单元测试 + `351` 个黑盒 CLI E2E 测试），`ruff check --select F,E9,W panflat tests` **`0` 错误**，`tests/e2e/` 零内部模块导入，`panflat_v1_final_revised.md` 规范文档零改动。
- **中断原因**：`gemini4 hydrogen extra` 子代理在完成 M7 Round 2 的全部 13 项代码修复（`R2-D1`..`R2-D13`）、新增 8 个单元测试（`test_m7_22`..`29`）及新增 6 个 CLI E2E 测试（`test_tier5_m7_09`..`14`）后，在同步测试计数文档与启动 Round 2 第 2 轮门禁复审前触发配额上限（`generic::resource_exhausted: Individual quota reached`）。
- **盘点范围**：结合已完成的 M7 Round 1/2 变更与针对 `panflat/*.py` 全模块的独立逐行代码审核及 `panflat_v1_final_revised.md` 规范逐条对账，剩余待完成工作汇总如下。

---

## 一、已完成的 M7 修复基线概览（已通过测试并推送）

在进入剩余任务前，以下模块级缺陷与硬化项**均已完成编码与回归验证**（无需重复开发）：

1. **`W` 行 `LazyStepTuple` 与四类图编码完全等价**（`panflat/gfa_io.py`、`panflat/normalizer.py`、`panflat/linearizer.py`、`panflat/centromere.py`）：修复了 `W` 行路径在紧凑模式下漏标 `multi_hit`、漏算 `DEL`/`CNV`、倒位与着丝粒锚点定位不一致，以及 `P`/`W` 精确同名 `src_path` 冲突未拦截的问题。
2. **PanSN 与非 PanSN 参考路径消歧及 `--backbone-prefix` 覆盖校验**（`panflat/gfa_io.py`、`panflat/verify_qc.py`）：强制 `#` 分隔符边界匹配，支持 `W` 行 `:start-end` 坐标后缀剥离、非 PanSN 精确匹配优先消歧，以及 `panflat merge` 后对非 PanSN / `W` 行路径显式 `--backbone-prefix` 覆盖校验。
3. **紧凑节点存储（`CompactNodeStore` / `CompactRefNodeSet`）安全边界**（`panflat/models.py`、`panflat/gfa_io.py`）：强制规范十进制正整数双射（拒绝前导零如 `01` 与 `> 50,000,000` 稀疏节点 ID 走紧凑数组），并补齐 `_extra` 回退与负编码绝对值边界检查。
4. **同位点 `CNV` + `DEL` 并存输出与多等位部分覆盖 `CNV` 分型**（`panflat/linearizer.py`）：移除了 `CNV` 对同位点 `DEL` 的静默抑制；修复了多等位 `CNV` 下单侧截断 contig 误赋具体等位号而非 `.` 的问题。
5. **两阶段 `--qc` 自比对屏蔽与连续参考比对**（`panflat/verify_qc.py`）：Stage 1 改为比对完整连续参考染色体序列（消除节点边界切碎导致的漏检），Stage 2 仅排除同一 `pan.fa` 物理区间自命中并将单 piece 目标裁剪至自身边界（消除跨等位 `N` spacer 嵌合比对）。
6. **多染色体结合律 `panflat merge` 与大文件校验加固**（`panflat/writers.py`、`panflat/verify_qc.py`）：支持扁平/左嵌套/右嵌套任意层级合并，`unresolved` 记录强制 `{chrom}_UNRES_` 前缀隔离，同步重映射 `ref.lifted.gff3` 中的 `sv_id`；补齐 `> 2 MB` `offsets.tsv` 快速路径的节点序列逐碱基校验、多链 `.chain` 块注释按链绑定，以及畸形 FASTA/GFA 严格拒绝。
7. **多父级（Multi-`Parent`）GFF3 平移与同源异构体独立校验**（`panflat/gff_liftover.py`、`panflat/verify_qc.py`）：支持逗号分隔多 `Parent` 共享 `exon`/`CDS` 的按转录本独立拆分与蛋白还原校验。

---

## 二、剩余编码任务（P1 — 独立代码审核发现的 4 项边界/契约尾项）

### 任务 1.1：修复多可变剪接转录本下 `gene` 容器特征的 `sv_status` 叠加聚合缺失
- **涉及文件**：`panflat/gff_liftover.py`（`liftover_gff3`，第 `553–601` 行）
- **规范依据**：`panflat_v1_final_revised.md` §4 第 4 条（第 `327–335` 行）——第 9 列 `sv_status` 标签**可叠加**（`intronic_insertion`、`utr_insertion`、`cds_split_by_insertion`）。
- **根因分析**：
  - 在修复 `R2-D12`（防止共享 `exon` 把兄弟转录本 `tx1` 的 `cds_split_by_insertion` 串染给仅作 UTR 使用的 `tx2`）时，代码删除了容器特征对子特征 `d.sv_statuses` 的继承，改为对所有容器特征统一遍历池化后的 `desc_cds` / `desc_utrs` / `desc_exons` 并执行互斥的 `if / elif / elif` 判断（第 `573–585` 行）。
  - 对于单个 `mRNA` 容器，任意插入点 `p` 在该转录本内确实互斥；但对于包含多个可变剪接 `mRNA` 的父级 `gene` 容器，同一插入点 `p` 在 `tx1` 中可落在 `CDS` 内（`cds_split_by_insertion`），而在 `tx2` 中可落在 `5'-UTR`（共享外显子非编码区，`utr_insertion`）或跳过的盒式外显子对应的内含子中（`intronic_insertion`）。池化 `if / elif / elif` 会因 `any(c.s0 < p < c.e0 for c in desc_cds)` 为真而短路，导致 `gene` 行丢失 `utr_insertion` 或 `intronic_insertion`。
- **修改方案**：
  1. 将容器特征标注拆为自底向上的两步：
     - **第一步（叶级容器，如 `mRNA` / `transcript` / `lnc_RNA`，即子特征中不含其他非可拆分容器）**：保持现有第 `568–585` 行的按自身 `desc_cds` / `desc_utrs` / `desc_exons` 评估逻辑，确保兄弟转录本之间绝不通过共享 `exon` 串染。
     - **第二步（上级容器，如 `gene`，即直接子特征中包含 `mRNA` / `transcript` 等容器特征）**：并集所有子容器已计算出的 `sv_statuses` 与 `sv_ids`（若 `gene` 下同时挂有不隶属任何子容器的直接 `exon`/`CDS` 或超出子容器范围的区间，再对该部分补充评估）。
  2. 保持最终输出顺序固定为 `("intronic_insertion", "utr_insertion", "cds_split_by_insertion")`。
- **验收测试**：
  - 单元测试（`tests/unit/test_m7_deep_review_hardening.py`）与黑盒 CLI E2E 测试（`tests/e2e/test_tier5_adversarial.py`）：
    1. 共享外显子场景：`tx1` 为 `CDS`、`tx2` 为 `UTR`，验证 `tx1` 仅含 `cds_split_by_insertion`、`tx2` 仅含 `utr_insertion`、父级 `gene` 同时含 `sv_status=utr_insertion,cds_split_by_insertion`。
    2. 盒式外显子跳跃场景：`tx1` 包含中间编码外显子（插入点落在其中）、`tx2` 跳过该外显子（同一插入点落在 `tx2` 内含子），验证父级 `gene` 同时含 `sv_status=intronic_insertion,cds_split_by_insertion`。

---

### 任务 1.2：隔离多等位 `INS` 事件 `alleles` 字段中的等位级标签（`partial_content=1` / `repeat_residual=1` / `multi_hit=1`）
- **涉及文件**：`panflat/linearizer.py`（`run_chromosome_pipeline`，第 `4293–4301` 行与第 `4353–4379` 行）
- **规范依据**：`panflat_v1_final_revised.md` §3.4.1（第 `120` 行）、§3.5（第 `154` 行）、§3.6.6（第 `208` 行）及 §5.1–§5.2（第 `369–383` 行）。
- **根因分析**：
  - 第 `4295–4300` 行已按每个等位签名 `sig_val` 分别记录了 `sig_has_partial[sig_val]`、`sig_has_rep_res[sig_val]`、`sig_has_mh[sig_val]`。
  - 但第 `4353–4361` 行又计算了整个平行分支组 `G(a,b)` 范围的 `group_has_partial`、`group_has_rep_res`、`group_has_mh`，并在第 `4371–4376` 行使用 `if sig_has_*(s) or group_has_*:` 附加到 `ordered_sigs` 中的**每一个等位**上。
  - 当同一锚点对 `(a,b)` 存在两个互不同源的插入等位（例如 `SampleA#1` 携带完整干净的等位 `1`，而 `SampleB#1` 携带触发 `repeat_guard` 部分屏蔽、`repeat_residual` 或 `multi_hit` 的等位 `2`；或者 `SampleB#1` 被 `repeat_guard` 完全屏蔽记为 `.`，仅剩 `SampleA#1` 的干净等位 `1`）时，干净的等位 `1` 也会被污染标上 `,partial_content=1` / `,repeat_residual=1` / `,multi_hit=1`。
- **修改方案**：
  1. 在构建 `sig_has_partial`、`sig_has_rep_res`、`sig_has_mh` 时，对每个等位签名 `s` 综合两部分证据：
     - 归属于签名 `s` 的原始保留候选集合 `sig_cand_ids[s]` 自身的 `partial_content` / `repeat_residual` / `multi_hit` 标志；
     - 签名 `s` 所选中的顶层代表 `r_idx in s[0]`（以及 `s[1]` 嵌套子组代表，若因相似聚类合并了带上述标志的同源成员）对应的代表级标志。
  2. 移除第 `4353–4361` 行及第 `4371–4376` 行跨异源等位无差别广播的 `or group_has_partial` / `or group_has_rep_res` / `or group_has_mh`。
- **验收测试**：
  - 构造同一 `(a,b)` 位点的双品系异源多等位 `INS`：品系 A 为 800 bp 干净单次命中插入（等位 1），品系 B 为含部分重复屏蔽残余或 `multi_hit` 的异源插入（等位 2），以及品系 B 全重复被完全屏蔽为 `.` 的对照场景；验证 `sv.tsv` 第 9 列 `alleles` 中等位 1 不含伪标签，等位 2 精确携带对应标签。

---

### 任务 1.3：强化着丝粒 `CEN_ISLAND` 挂载点局部偏移差的精确等式校验（规范 §3.8.5）
- **涉及文件**：`panflat/verify_qc.py`（`verify_build_outputs`，第 `2926–2966` 行）
- **规范依据**：`panflat_v1_final_revised.md` §3.8.4-D（第 `303–305` 行）与 §3.8.5（第 `314` 行）：
  > “在每个挂载点比较插入前后的局部offset差，必须等于本节代表总长、同位置普通插入总长与统一布局后实际spacer总长之和；不能把上游其他插入造成的累计offset算进来。”
- **根因分析**：
  - 当前 `verify_qc.py` 第 `2961` 行仅校验了下界不等式 `slot_ins_len < cen_bp + 2 * eff_spacer_n`。
  - 若挂载点处发生正向膨胀篡改（例如 `CEN_ISLAND` 与同位置普通 `INS` 共挂载时违反 §3.8.4-D“相邻插入块之间使用一个 spacer，不叠加重复 spacer”而多插入了一段 `N * spacer_n`，或槽位内混入未在 `inserts.tsv` 登记的碱基），下界不等式无法检出。
- **修改方案**：
  1. 在 `verify_qc.py` 第 `2926–2966` 行获取挂载点 `r_e` 对应的精确槽位区间 `[slot_pan_start, slot_pan_end)`（长度为 `slot_ins_len`）。
  2. 收集落在该槽位内的全部 `CEN_ISLAND` piece（总长 `cen_bp`）与同位置普通 `INS` piece（总长 `normal_ins_bp`），验证各 piece 在槽位内互不重叠，且槽位内除 piece 以外的所有间隙均由 `N` 组成。
  3. 按 §3.8.4-D 与 §3.6.4 统一布局规则计算实际 spacer 总长 `actual_spacer_bp = slot_ins_len - cen_bp - normal_ins_bp`（并校验相邻 piece / 边界之间的每段连续 `N` 长度恰好等于 `eff_spacer_n`，不存在 `2 * eff_spacer_n` 重复叠加；若无普通 `INS` 共挂载，则尾随 `v` 之前也必须恰好有 `1 * eff_spacer_n`），强制断言 `slot_ins_len == cen_bp + normal_ins_bp + actual_spacer_bp` 及单段 spacer 精确性。
- **验收测试**：
  - 在 `tests/unit/test_m7_deep_review_hardening.py` 与 `tests/e2e/test_tier5_adversarial.py` 中新增 C14 共挂载点插入双倍 `spacer_n` 的篡改用例，验证 `panflat verify` 稳定拦截并以退出码 `2` 报错。

---

### 任务 1.4：对齐 `unresolved.fa` FASTA Header 的独立 `strand=<±>` 字段格式与校验
- **涉及文件**：
  - `panflat/writers.py`（`write_build_outputs` 第 `296–300` 行；`merge_chromosomes` 第 `851–855` 行）
  - `panflat/verify_qc.py`（`verify_build_outputs` 第 `430–485` 行与第 `2668–2680` 行）
- **规范依据**：`panflat_v1_final_revised.md` §5 输出文件表（第 `354` 行）：
  > `{p}.unresolved.fa`：仅包含5.5规定需导出序列的条目；header：`>id reason=<UNANCHORED_END|LARGE_JUMP|INV_EDGE|INV_CONFLICT|CEN_*> src=<路径>:<起>-<止> strand=<±>`
- **根因分析**：
  - 当前 `writers.py` 第 `297–299` 行与第 `852–855` 行写出的 header 为：
    `>{id} region_id={region_id} event_id={event_id} src={src_path}:{src_start}-{src_end}({src_strand}) len={length} reason={reason}`
    把链方向嵌在了 `src=...(+)` 圆括号中，缺少规范明确列出的独立 `strand=<±>` 键值字段。
  - 同时 `verify_qc.py` 读取 `unresolved.fa` 时只保留了 `seq_name -> sequence`，未校验 header 中的 `reason=`、`src=`、`strand=` 与 `unresolved.tsv` 对应记录是否一致。
- **修改方案**：
  1. 将 `writers.py` 单染色体输出与 `merge_chromosomes` 合并输出两处的 `unresolved.fa` header 统一为包含规范必选键值字段的格式：
     `>{id} reason={reason} src={src_path}:{src_start}-{src_end} strand={src_strand} region_id={region_id} event_id={event_id} len={length}`
  2. 在 `verify_qc.py` 解析 `unresolved.fa` 时保留完整 header 键值映射，并在第 `2668–2680` 行核对 `reason`、`src`（`<src_path>:<src_start>-<src_end>`）与 `strand`（`+`/`-`）与引用该 `exported_fasta_id` 的 `unresolved.tsv` 行一致（注意 §5.5 允许同一源区间与方向去重共享同一条 FASTA 记录，若多个 `reason` 关联同一 `exported_fasta_id`，核对 `src` 与 `strand` 完全一致且 `reason` 属于关联集合）。
- **验收测试**：
  - 更新/新增单元测试与 CLI E2E 断言，检查生成的 `unresolved.fa`（构建产物与 `merge` 产物）header 含独立 `reason=...`、`src=<path>:<start>-<end>`、`strand=<+|->`，且篡改 header 中 `strand` 或 `src` 时 `panflat verify` 能够检出。

---

## 三、文档与测试元数据同步任务（P2 — 配额中断遗留）

### 任务 2.1：同步 `TEST_READY.md` 与 `PROJECT.md` 的测试统计数据
- **涉及文件**：
  - `TEST_READY.md`（第 `3–27` 行、第 `32–36` 行）
  - `PROJECT.md`（第 `109` 行）
- **现状说明**：
  - 当前代码库实际收集并通过的测试数为 **`517` 个**（单元测试 **`166`** 个 + E2E 测试 **`351`** 个）：
    - `tests/unit/test_m1_foundation.py`：`34`
    - `tests/unit/test_m2_normalizer_centromere.py`：`44`
    - `tests/unit/test_m3_linearizer.py`：`49`
    - `tests/unit/test_m4_liftover_writers_verify_cli.py`：`10`
    - `tests/unit/test_m7_deep_review_hardening.py`：`29`
    - `tests/e2e/test_tier1_features.py`：`133`
    - `tests/e2e/test_tier2_boundaries.py`：`132`
    - `tests/e2e/test_tier3_combinations.py`：`28`
    - `tests/e2e/test_tier4_scenarios.py`：`14`
    - `tests/e2e/test_tier5_adversarial.py`：`44`
  - 但 `TEST_READY.md` 仍停留在 Round 1 快照（写为 `345` E2E + `158` Unit = `503`），`PROJECT.md` 第 `109` 行停留在 `worker_m7_r2_1` 快照（写为 `166` unit + `345` E2E = `511/511`）。
- **执行动作**：
  - 在完成上述 P1 编码任务及新增回归测试后，按 `pytest --collect-only -q tests/unit tests/e2e` 的实测精确数字一次性同步更新 `TEST_READY.md` 与 `PROJECT.md`。

---

## 四、门禁复审闭环与真实规模数据回归冒烟（P3）

### 任务 3.1：执行全量静态检查、黑盒纯度检查与回归门禁复审
- **执行命令**：
  ```bash
  # 1. 代码静态检查（零错误）
  ruff check --select F,E9,W panflat tests

  # 2. E2E 黑盒纯度检查（tests/e2e 下零 panflat 内部导入）
  ! rg -n '^\s*(import|from)\s+panflat\b' tests/e2e

  # 3. 规范文件零改动核查
  git diff --exit-code panflat_v1_final_revised.md

  # 4. 全量单元测试 + CLI E2E 测试套件
  pytest -q
  ```
- **完成标准**：上述四项检查全部 `exit 0`，并对 P1 四项补丁完成独立代码复审确认无回归。

### 任务 3.2：真实染色体规模数据（§8.3 / M6 规模）端到端冒烟复核
- **说明**：在挂载真实植物泛基因组染色体级 GFA 数据集（如 35 单倍型 `Pyrar_xinyeli` Chr10A 2.15 GB GFA，或 ~30 单倍型水稻 Chr01，并配齐 `--ref-fa`、`--ref-gff`、可选 `--centromere-bed`）的环境中执行：
  1. 单染色体及多染色体 `panflat build --qc --threads 16`；
  2. `panflat merge` 合并多染色体输出；
  3. 独立运行 `panflat verify`（含显式 `--backbone-prefix` 与 `--ref-gff`），确认不变量 I/II/III、着丝粒 5 路碱基守恒与挂载点精确偏移差、GFF3 转录本与翻译蛋白序列一致性全部通过，且运行时间与峰值内存符合 §8.3 目标（16 线程 ≤ 2 小时，峰值内存 ≤ 32 GB）。
