# 分析方案：Style Drift 实验 LLM-as-Judge 验证与主分析

**数据来源**：`stress_test_100turn_deepseek/output_aggregate_20260322/results_aggregate_success_20260322.jsonl`
（15,555条有效行，3条件 × 4域 × 52对话 × 100轮）

---

## 总体流程

```
原始数据 (15,555行, 1-10/1-5量表)
    │
    ▼
[Step 1] 重新打分 — 用 judge_system_v2.txt 对全量数据跑 LLM4
    │    输出: data_judged_1_7.jsonl (15,555行, 1-7量表)
    │
    ▼
[Step 2] 构建标注集 — 分层抽样 36条
    │    输出: calibration_set_36.jsonl
    │
    ▼
[Step 3] 人工标注 — 5位标注者对 36条打 1-7分
    │    计算 inter-rater ICC
    │
    ▼
[Step 4] LLM-Human 一致性验证
    │    LLM4分 vs 人工分 → Pearson r (每个维度)
    │    目标: r > 0.7
    │
    ▼
[Step 5] 主要分析 — 风格漂移模式
         输出: 图表 + 统计结果
```

---

## Step 1 · LLM4 重新打分（1-7量表）

**为什么重跑而非转换旧分数**：
旧分数使用 1-10 量表的 prompt（不同锚点定义），线性转换数学上成立但概念上存在量表不等价问题。用新 `judge_system_v2.txt` 直接打 1-7，分数可直接与人工标注（也是1-7）对比，无需额外justify转换逻辑（参考 G-Eval, Liu et al., 2023 的做法：judge prompt 和 human annotation 使用同一量表）。

**输入**：`results_aggregate_success_20260322.jsonl`（只取 `model_reply` 和 `user_text` 字段）
**脚本**：`stress_test_150turn_deepseek/run_judge_parallel.py`（已有）
**配置**：`judge_system` 替换为 `annotation/judge_system_v2.txt`
**并发**：`--workers 20`
**估计成本**：15,555次调用 × ~500 tokens ≈ $15–30（取决于LLM4选型）
**输出**：`analysis/data_judged_1_7.jsonl`

---

## Step 2 · 构建标注集

**分层策略**：
```
3条件 (default/cynical/distant)
× 4域 (creative/finance/mental_health/medicine)
× 3epoch (early:turn1-33 / mid:34-67 / late:68-100)
= 36格，每格随机抽1条
→ 36条
```

**文献依据**：Koo & Mae (2016) 建议 ICC 可靠估计至少30个subject，36条满足下限。

**脚本**：`annotation/build_calibration_set.py`（去掉 supplement 参数，只用主数据）
**输出**：`annotation/calibration_set_36.jsonl`（含 LLM4 的1-7分，用于 Step 4 对比）

---

## Step 3 · 人工标注

**标注工具**：`annotation/calibration_set_36_for_annotation.csv`（Google Sheets 分发）
**标注指南**：`annotation/annotation_guide.md`（1-7量表，9个维度，含3个校准例子）

**标注者分工**：
```
标注者 A + B：第 1–18条（+ E 仲裁）
标注者 C + D：第 19–36条（+ E 仲裁）
标注者 E    ：全部 36条
```

**标注者负担**：36条 × 9维度 = 324个评分/人（A/B/C/D）；E = 648个评分

**一致性计算**：
- 组内相关系数 **ICC(2,1)**（双向随机，单次测量）（Shrout & Fleiss, 1979）
- 判断标准：ICC < 0.5 差；0.5–0.75 中等；0.75–0.9 好；> 0.9 优秀（Koo & Mae, 2016）

**分歧处理**（参考 annotation_guide.md）：
```
差距 ≤ 1分 → 取均值
差距 = 2分 → 标注者E仲裁
差距 ≥ 3分 → 小组讨论取共识
```

---

## Step 4 · LLM-Human 一致性验证

**对比**：36条的 LLM4 1-7分 vs 人工平均分（解决分歧后）

**指标**：
- 每个维度的 Pearson r 和 Spearman ρ
- 总体 Macro-average r

**判断标准**（参考 Prometheus, Kim et al., 2023；G-Eval, Liu et al., 2023）：
```
r > 0.7  → LLM judge 可信，Step 5 正常进行
r 0.5–0.7 → 可接受，论文中注明局限性
r < 0.5  → 需要重新审视 judge prompt
```

**脚本**：`annotation/compute_agreement.py`（待写）
**输出**：各维度 r 值表 + 散点图

---

## Step 5 · 主要分析（风格漂移）

**数据**：`analysis/data_judged_1_7.jsonl`（15,555行）

**分析一：条件间差异**
- 三个条件（default/cynical/distant）在9个维度上的均值对比
- 统计检验：单向 ANOVA + Tukey HSD 事后检验

**分析二：轮次趋势（核心研究问题）**
- 每个条件下，9个维度随 turn_index 的变化趋势
- 方法：按 epoch（early/mid/late）分组均值 + 线性回归斜率
- 是否存在向 default 收敛的"均值回归"？

**分析三：域差异**
- 4个域（creative/finance/mental_health/medicine）在各维度的表现差异
- 某些域是否对风格漂移更敏感？

**脚本**：`analysis/style_drift_analysis.py`（待写）
**输出**：
- 各条件 × 维度均值表
- 轮次趋势折线图（9个维度 × 3个条件）
- ANOVA 结果表

---

## 文件清单

| 文件 | 状态 |
|------|------|
| `annotation/judge_system_v2.txt` | ✅ 已完成 |
| `annotation/annotation_guide.md` | ✅ 已完成 |
| `annotation/build_calibration_set.py` | 需修改（去掉supplement） |
| `stress_test_150turn_deepseek/run_judge_parallel.py` | ✅ 已有，需调整config |
| `annotation/compute_agreement.py` | ❌ 待写 |
| `analysis/style_drift_analysis.py` | ❌ 待写 |

---

## 参考文献

- Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. *EMNLP 2023*.
- Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*.
- Kim et al. (2023). Prometheus: Inducing Fine-grained Evaluation Capability in Language Models. *ICLR 2024*.
- Shrout & Fleiss (1979). Intraclass correlations: Uses in assessing rater reliability. *Psychological Bulletin*.
- Koo & Mae (2016). A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research. *Journal of Chiropractic Medicine*.
- Bartneck et al. (2009). Measurement Instruments for the Anthropomorphism, Animacy, Likeability, Perceived Intelligence, and Perceived Safety of Robots. *IJES*.
