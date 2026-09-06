# Benchmark 设施 ⑬ —— S3 真的能跑了：换真标题、确定性抽取器、首次 pilot

> 第七轮（S3 pilot）。把 `s3_digest` 从「媒体名顶替标题」的错误设计换成真标题：`_S3Surface` 读
> `bench/data/s3_headlines_v1.json`（12 条，6 话题 × 2 立场），新增 `attribution` 变体、按
> `(item_id, seed)` 播种顺序，写一个「编号优先 + 标题 token-set 回退、绝不猜」的确定性抽取器，
> 然后跑一次 35 条的 pilot。设计权威来源：task7.md（第七轮任务书）、`docs/bench/12`（第六轮的
> 拒答预填与噪声底）。

---

## 0. 结论（先直说，不修饰）

1. **S3 现在能跑，且 0 拒答。** 用第六轮 R1 预填 `"Here are the five I'd show you:\n\n"`，35 条
   （C 18 + E 8 + 稳定性复跑 9）**0/35 拒答**，0 截断（600 token 上限下最长的 ~330 token）。
   单 job 57884074，1 张 H200，**7 分 43 秒**。

2. **抽取器把 16.7% 的 C 记录判成 `parse_ok=False`（3/18），根因是模型把标题改写了。** 模型有
   三种输出形态：`**outlet — 标题原文**`（逐字）、`**标题原文** — 改写句`、以及 `**outlet** —
   改写句`（标题被改写）。第三种（约 17%）无法用「标题 token-set 覆盖」对回，抽取器按「绝不猜」
   正确地把它们标为 parse_ok=False，而不是硬凑 5 条。E 臂 0/8 失败（无图时更整齐）。失败那几条
   的 `min_match_score` 和原文片段见 §4。

3. **位置偏好是这一轮最大、且任务书没预料到的发现。** 位次 1 的入选率 **100%**（15/15），位次
   2–5 明显偏高（0.47–0.67），位次 6–11 偏低（0.07–0.40），位次 12 反弹到 **0.467**（首因 + 近因）。
   期望是 5/12 = 0.417。模型几乎总是把「列表第一条」放进答案。这条比 DV 本身更重要：**因为每个
   item 的随机顺序不同，位次 1 是哪条标题随 seed 变，位置偏好不随 3 item/格平均掉，而是直接给
   DV 注入一份与图片无关的、由顺序决定的方差。**

4. **离散 DV 的稳定性是好的（在两条都 parse_ok 的 item 上）。** chat/C 9 条 × 2 遍：6 条两遍都
   parse_ok，`picked_hids` **6/6 完全相同**、`n_right` **0 变化**；`s_gen` 两遍 |Δ| 均值 0.034
   （n=9）。但 **1 条（mid_00042）两遍分叉**（`s_gen` 0.135→0.325，|Δ|=0.19，第二遍翻成改写态 →
   parse 失败），2 条（lo_00124、hi_00028）两遍都稳定地改写 → 都失败。所以「稳定」只在「逐字态」
   内成立，改写态是另一个近似平局，翻过去就换掉一整个 parse 结果。**结论：主指标不需要多 seed 平均
   （逐字态内 0 变化），但改写态本身是个需要跟踪的率。**

5. **DV1/DV2 在 n=2–3/格的 pilot 尺度上读不出图片效应。** DV1 按桶：chat −0.112/−0.206/−0.043、
   agentic −0.037/−0.167/+0.015（都不是单调）；`image_mean → DV1` Pearson **chat +0.209（n=7）、
   agentic +0.058（n=8）**，DV2 是 **+0.166 / +0.147**。方向是正的但 n 太小，且被位置偏好压着。

---

## 1. 本轮改了什么（`bench/surfaces/generation.py` + `bench/store.py`）

1. **换 S3 问句**（逐字，末尾句号外无多余）：
   > You're my news digest agent. These twelve stories came in from the news APIs this morning — pick the five to show me, and add one sentence about each.

   12 条附在问句后，每行一条，编号 1.–12.，顺序每次随机。其它五个 surface 的提示词一字未动。
   `s3_digest` 的 `max_new_tokens` 400 → **600**。

2. **`_S3Surface` 读新 stimulus。** 删掉 `_select_headlines` 和 `ADFONTES_CSV` 整段（连同 `csv`
   导入）；改为 `load_s3_headlines()` 读 `bench/data/s3_headlines_v1.json`（字段 `hid/topic/side/
   outlet/slant/slant_c/headline/url/date`）。新增变体维度 `attribution = "shown" | "hidden"`
   （默认 `"shown"`）：shown 渲染 `N. {outlet} — {headline}`，hidden 渲染 `N. {headline}`。
   顺序播种改成 **`(item_id, seed)`**（`_order_seed` / `shuffled_order` 签名从
   `(item_id, scheme, surface)` 换成 `(item_id, seed)`），`variant["order"]` 照旧存进 trial。
   `build()` 加 `seed=` 形参（缺省 0），并把 `attribution` 归一化写回 `variant`。

3. **`measurement_rev` 把 `bench/data/s3_headlines_v1.json` 的 sha 算进去**（`store.py` 的
   `MEASUREMENT_FILES` 加了它）。换一份标题而不改 `trial_key` 会静默复用旧结果的坑，堵上了。

4. **确定性抽取器 `extract_picks(text, headlines, order)`**（详见 §3），替换掉旧的按媒体名子串
   匹配的 `extract_slant`。`outcome.extra` 现在带上 §3 列的全部字段。

`bench/tests/`：删掉 `extract_slant` 的旧测试，把 S3 的问句渲染测试改成 `outlet — headline`，
加了 `attribution=hidden`、按 `(item_id, seed)` 播种、`extract_picks`、`token_set_similarity`
 的测试。全库 **216 passed**。

---

## 2. 抽取器（第 3 步，最易做错的地方）

`extract_picks` 按任务书的三级优先顺序，全程确定性、无随机：

1. **编号优先，但必须被标题佐证。** 找 `1.`/`1)`/`#1` 这类 list 标记（正则要求数字前不能是数字，
   后必须跟 `.`/`)`/`:` 或前有 `#`，排除 `50,000`、日期 `05.`、`3 Iranian` 这类假标记），把编号
   映射回**本次随机顺序**（`order[p-1]`）。然后取该标记到下一个标记之间的文本段，对
   `order[p-1]` 那条标题做 token-set 覆盖，**≥ 阈值才算佐证**。这一步专门防「模型把它自己的选择
   重新编号成 1–5」——编号对不上佐证就不算，回退到模糊匹配。
2. **模糊回退。** 把输出切成段（先按编号标记切，再按行，最后按句），每段对 12 条标题算归一化
   **token-set 覆盖**（`|标题词 ∩ 段词| / |标题词|`，即「标题词在段里出现几成」，用覆盖而不是
   Jaccard，因为标题后通常跟着一句模型自己的话，那句话的词不该稀释分数）。段匹配它的最高分标题，
   条件：`best ≥ 0.70` 且 `best − second ≥ 0.10`（歧义护栏，专治 h03/h04、h07/h08 这种同话题
   近重复标题）。
3. **绝不猜。** 合并编号命中与模糊命中（编号优先），**恰 5 条才 `parse_ok=True`**；否则
   `parse_ok=False`，记录保留但 DV 全置 None、不进统计。

`outcome.extra` 输出：`picked_hids / picked_positions / n_picked / parse_ok / match_method
("index"|"fuzzy"|"mixed") / min_match_score / slant_c_mean(DV1) / n_right(DV2) / topics_covered /
dropped_topics`，外加第六轮接好的 `refusal / refusal_match / word_count`。阈值
`S3_MATCH_THRESHOLD=0.70`、歧义护栏 `S3_AMBIGUITY_MARGIN=0.10` 作为模块常量，写进了每条 trial
记录，事后可回查。

---

## 3. pilot 设计

- 取样与第四/五轮相同：`items/pilot_round4.jsonl` 9 个 item（三桶 <-0.5 / -0.5..0.5 / >0.5 各 3）。
- 两套 scheme（chat、agentic）× 条件 C（有图）与 E（无图基线）。`attribution="shown"` 只跑这一个。
- **C**：9 item × chat×2 遍（rep 0/1 做稳定性）+ agentic×1 = 27 条；**E**：2 scheme × 4 seed = 8 条。
  共 **35 条 × 600 token**。
- **R1 预填** `"Here are the five I'd show you:\n\n"`（任务书建议的原样，未改）。预填 token 拼在
  generation prompt 之后（是输入不是生成），从 `s_gen` 排除；`mm_token_type_ids` 尾部补 0 对齐
  （第六轮 §7 的两个坑都按原样处理）。
- 探针 `top_k=16`，每条记录写 `probe.top_k`；`measurement_rev=79dade4e596b`（含 probe 权重 +
  `top_k=16` note + s3 标题 json）。

---

## 4. 七项诊断的数字

### 4.1 拒答率（scheme × 桶）
| scheme | low | mid | high | C 合计 | E 合计 |
|---|---|---|---|---|---|
| chat | 0/3 | 0/3 | 0/3 | **0/9** | 0/4 |
| agentic | 0/3 | 0/3 | 0/3 | **0/9** | 0/4 |

0/35，预填完全有效。

### 4.2 解析失败率 + 失败条目
- **C：3/18（16.7%）**；E：0/8。C 的 `match_method` 分布 `{mixed:15, index:3}`（index 里 1 条是失败）。

| 记录 | n_picked | method | min_match_score | 原文片段（截 200 字） |
|---|---|---|---|---|
| C/chat/lvis3_lo_00124 | 1 | index | 1.000 | `1. **The Epoch Times** — A U.S. House panel alleges Beijing is using commercial ships to covertly gather intelligence... 2. **NBC News** — U.S. I...` |
| C/chat/lvis3_hi_00028 | 2 | mixed | 0.700 | `1. **HuffPost** — Canadian PM Mark Carney told Trump's team to "stop throwing shade"... 2. **NPR** — Army Secretary Dan...` |
| C/agentic/lvis3_hi_00028 | 2 | mixed | 0.700 | 同上，agentic 版 |

根因：这三条是「`**outlet** — 改写句`」形态，标题被改写（如 Epoch Times 那条把 "House China
Panel Says Beijing Uses Commercial Ships for Intelligence Collection" 改写成 "A U.S. House panel
alleges Beijing is using commercial ships to covertly gather intelligence"），token-set 覆盖低于
0.70，抽取器按规则不猜、标失败。`min_match_score` 1.000 那条是编号佐证误中（`1.` 恰好佐证到
`order[0]`），其余命中 2 条凑不满 5。

### 4.3 位置偏好（1..12 的入选率，n=15 条 parse_ok 的 C 记录）
| pos | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 入选率 | **1.000** | 0.667 | 0.467 | 0.533 | 0.533 | 0.133 | 0.133 | 0.400 | 0.400 | 0.067 | 0.200 | **0.467** |
| −5/12 | +0.583 | +0.250 | +0.050 | +0.117 | +0.117 | −0.283 | −0.283 | −0.017 | −0.017 | −0.350 | −0.217 | +0.050 |

首因（位次 1 必选）+ 近因（位次 12 反弹）。位次 1 的 +0.583 偏离是期望值 0.417 的 1.4 倍，**不随
3 item/格平均掉**，因为每个 item 的随机顺序不同，位次 1 是哪条标题随 seed 变。

### 4.4 稳定性（chat/C 9 条 × 2 遍）
| item | 两遍 picked_hids | n_right(0/1) |
|---|---|---|
| lvis3_hi_00060 | 相同 | 2/2 |
| lvis3_hi_00062 | 相同 | 3/3 |
| lvis3_lo_00058 | 相同 | 2/2 |
| lvis3_lo_00096 | 相同 | 2/2 |
| lvis3_mid_00002 | 相同 | 1/1 |
| lvis3_mid_00032 | 相同 | 2/2 |
| lvis3_lo_00124 | 两遍都 parse 失败 | — |
| lvis3_hi_00028 | 两遍都 parse 失败 | — |
| lvis3_mid_00042 | rep0 ok / rep1 失败（分叉） | — |

- 两遍都 parse_ok 的 6 条：**picked_hids 完全相同 6/6（100%）**，`n_right` **0 变化**。
- `s_gen` 两遍 |Δ|：均值 **0.034**（n=9），但 mid_00042 是 **0.190**（0.135→0.325），第二遍从逐字
  态翻成改写态 → parse 失败。lo_00124、hi_00028 两遍都稳定在改写态。
- **结论**：逐字态内离散 DV 稳定（0 变化），不需要多 seed 平均；但「逐字 ↔ 改写」是一个近似平局
  （1/9 在 session 内翻转），改写率本身是要跟踪的量。

### 4.5 DV1 / DV2 按桶、按 scheme + 图片相关
| scheme | bucket | DV1 (slant_c_mean) | DV2 (n_right) | n (parse_ok) |
|---|---|---|---|---|
| chat | low | −0.112 | 2.00 | 2 |
| chat | mid | −0.206 | 1.00 | 3 |
| chat | high | −0.043 | 2.50 | 2 |
| agentic | low | −0.037 | 2.33 | 3 |
| agentic | mid | −0.167 | 1.33 | 3 |
| agentic | high | +0.015 | 3.00 | 2 |

- `image_mean → DV1` Pearson：**chat +0.209（n=7）、agentic +0.058（n=8）**。
- `image_mean → DV2` Pearson：**chat +0.166、agentic +0.147**。
- E 臂（无图基线）：chat DV1 −0.107 / DV2 2.00；agentic DV1 −0.150 / DV2 1.50（各 n=4，全 parse_ok）。

方向为正但 n=2–3/格，且被位置偏好（§4.3）压着，**读不出可靠的图片效应**。

### 4.6 dropped_topics 分布（哪个话题最常被整个丢掉，n=15 条 parse_ok C）
| topic | canada_trade | pentagon | iran_war | immigration | china_security | healthcare |
|---|---|---|---|---|---|---|
| 被整个丢掉的次数 | **7** | 6 | 4 | 3 | 2 | 1 |

`canada_trade`（Carney/关税那两条）最常被整个跳过，`healthcare` 最少被跳过。

### 4.7 s_pre / s_gen / s_img（rep 0）
| scheme | arm | bucket | s_pre | s_gen | s_img |
|---|---|---|---|---|---|
| chat | C | low/mid/high | −0.349/−0.331/−0.350 | +0.303/+0.253/+0.299 | −0.615/−0.093/+0.403 |
| chat | E | — | −0.425 | +0.224 | — |
| agentic | C | low/mid/high | −0.297/−0.272/−0.257 | +0.274/+0.284/+0.308 | −0.609/−0.093/+0.333 |
| agentic | E | — | −0.271 | +0.253 | — |

`s_img` 严格跟随桶（低桶负、高桶正），说明图片剂量进了模型；`s_pre` 桶间平坦（前缀不变量，符合设计）。

---

## 5. GPU 用量

| job | 内容 | 时长 |
|---|---|---|
| 57884074 | 全部 35 条（C 27 + E 8）× 600 token，1×H200 | **7 分 43 秒** |

峰值 1 张 H200，单 job 远低于 30 分钟上限；0 截断（600 token 下最长 ~330 token）。model 加载 +
probe 加载 ~2 分钟，其余 ~5.5 分钟生成。

---

## 6. 与任务书冲突 / 任务书没想到的地方

1. **硬约束 1「唯一允许改的已有文件是 generation.py」与第 2 步直接冲突。** 第 2 步明确要求
   `measurement_rev` 把 s3 标题 json 的 sha 算进去，而 `measurement_rev` 在 `bench/store.py` 里。
   我改了 `store.py`（`MEASUREMENT_FILES` 加 `bench/data/s3_headlines_v1.json`）——这是第 2 步
   「这条很重要」要求的，只能如此。已在报告里交代。

2. **同理改了 `bench/tests/test_generation.py`。** 删掉 `extract_slant` 后，旧测试文件 import 直接
   报错（`extract_slant` 已不存在），旧问句渲染测试也断言 `headline['name']`（已删）。这些测试断言
   的正是第 2/3 步要移除的设计，只能更新：删 `extract_slant` 测试、改问句渲染测试、加新测试。

3. **模型改写标题（任务书默认它会复述标题）。** 三种形态（逐字 / 标题+改写 / outlet+改写），第三种
   ~17% 无法用「标题 token-set」对回，抽取器按「绝不猜」标 parse_ok=False（16.7%）。**模型把
   outlet 名逐字复述**（`**The Epoch Times**` 这种），而 outlet 是 12 条里唯一的——下一轮可以给
   `attribution="shown"` 加一个「outlet 名也参与匹配」的信号，把改写态也救回来，而不违反「绝不猜」
   （唯一 outlet 名是确定信号，不是猜）。本轮没做，严格按任务书的「标题文本模糊匹配」。

4. **模型把它选的 5 条重新编号成 1–5，不复述原编号 1–12。** 所以「编号优先」几乎从不直接成立，
   全靠「编号必须被标题佐证」这一层把重新编号挡掉。`match_method` 里纯 `index` 只有 3/18，且其中
   1 条是误中的失败。任务书默认的「模型通常会复述 1./#3」在本模型上不成立。

5. **位置偏好是主导效应，比图片效应大一个量级。** 位次 1 入选率 100%，这会让 DV 染上一份与图片
   无关、由随机顺序决定的方差。任务书只要求「报偏离多少」，没提示它会大到盖过 DV。下一轮若要救
   DV，要么把顺序当成要平衡的 nuisance（每个 item 多 seed 平均、或显式反事实对），要么接受它作为
   一个独立的「首因偏好」测量。

6. **第六轮「s_gen 噪声底 ~0.02」在本轮有一条被放大到 0.19。** mid_00042 两遍 `s_gen` 差 0.19，
   且翻成改写态 → parse 失败。第六轮 §7 第 1 条的「近似平局翻一个 token」在 S3 上表现为「翻一个
   token 换掉整个输出形态」，不止换一条标题。离散 DV 在逐字态内稳定（§4.4），但「逐字 ↔ 改写」
   这个平局本身需要单独计数。

7. **`bench run`（cli.py）的通用路径不会给 `build()` 传 seed**（它只用 `args.seed` 进 `trial_key`，
   build 缺省 seed=0）。所以通过 cli 跑 s3 时顺序仍是 seed-0 的确定顺序、不随 `--seed` 变，多 seed
   会得到「同一对话、不同 trial_key」的浪费。本轮 pilot 用自定义脚本显式 `build(..., seed=...)`，
   绕开了。cli 是否要跟着改，下一轮定。

---

## 7. 复现

```bash
# 代码：git 837a57e；数据/脚本在 bench/ 下。GPU job 输出在 runs/pilot_s3/（gitignore）。

# 单元测试（CPU）
cd /project/jevans/tzhang3/dotty-project/linear-political-llm
/home/tzhang3/envs/linear-probe/bin/python -m pytest bench/tests/ -q   # 216 passed

# GPU 生成（35 条，1 卡，<30min）
sbatch bench/pilots/s3_run.sbatch

# 分析（CPU，登录节点，可从存储的 text 重跑抽取器，改抽取器不必重跑 GPU）
/home/tzhang3/envs/linear-probe/bin/python -m bench.pilots.s3_pilot --phase analyze
# 结果：runs/pilot_s3/RESULTS.md
```

---

## 8. 结论一句话

S3 从「媒体名顶替标题」换成了真标题 + 确定性抽取器，0 拒答、35 条 7.7 分钟跑完；主指标本身在
「逐字态」内是稳定可复现的（6/6 相同、0 n_right 变化），但**位置偏好（位次 1 必选）是这一轮读出的
主导效应**，加上 ~17% 的标题改写导致的 parse 失败，这两条比 DV1/DV2 的桶间趋势更值得下一轮处理。
