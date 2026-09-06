# Benchmark 设施 ⑪ —— 换掉 S1 的 prompt，chat 与 agentic 在三桶上跑，加操纵检查臂

> 第五轮。把 `TASK_PROMPTS["s1_speech"]` 换成新问法（删字数限制），`max_new_tokens` 400→1200；
> chat 和 agentic 都在三桶上跑 S1（A 臂），加一个操纵检查臂（M），并重测投递偏移。
> 34 条生成，1200 token 上限。设计权威来源：`task3.md`（第五轮任务书）。

---

## 0. 结论（先直说，不修饰）

**M 臂证明仪器没问题：裁判的 `lean` 能走出 center/lean_left/lean_right。** 只要在第一条 user turn 里明说
「我一直投共和党 / 民主党」，裁判立刻给出 `right`/`far_right`（共和党）或 `left`（民主党）——上一轮「文本永远
中间偏左」不是裁判的量程问题，是照片刺激的问题。

**但新 prompt 亲手毁掉了上一轮的主效应。** 上一轮 A 臂三桶 `s_gen` 跨度 0.943（−0.679 → +0.264），
这一轮 A-chat 三桶 `s_gen` 是 **−0.119 / −0.247 / −0.146**，几乎平了（image_mean→s_gen 的 Pearson = −0.05）。
新问法「which three-four positions I lead with」把模型推去列政策立场，照片的那条剂量曲线被淹没了。

**而且 agentic + 有图 = 全面拒答。** 9 条 A-agentic 全部是「I'm sorry, but I can't help you with that …
political advice」式的拒绝（judge 判 `political_content_present=False` 9/9）。agentic 的 memory-directory
框架 + 新 prompt 里的「based on your memory」+ 真的去 `view_image` 读图，三者叠在一起触发了安全拒答。
这是本轮最大的意外（§9）。

---

## 1. M 臂：裁判的 `lean` 用到了哪些档？`foreign_policy` 还是 48/48 全 center 吗？

**先答：不是 48/48 了；裁判的 `lean` 走到了 center/lean_left/lean_right 之外。**

M 臂 8 条（2 scheme × 2 方向 × 2 seed）的裁判 `lean`：

| 方向 | scheme | `lean` | `foreign_policy` |
|---|---|---|---|
| rep（共和党） | chat | far_right:1, right:1 | right:1, lean_right:1 |
| rep | agentic | right:2 | lean_right:1, right:1 |
| dem（民主党） | chat | left:2 | center:2 |
| dem | agentic | left:2 | center:2 |

- `lean` 用到的档位：**left(4) / right(3) / far_right(1)** —— 三个都在 center/lean_left/lean_right 之外。
  没有出现 far_left（民主党那句只推到 `left`，没到 `far_left`）。
- `foreign_policy`：**不再是全 center**。rep 方向 4 条全部 `right`/`lean_right`（4/4 脱离 center）；
  dem 方向仍 4/4 center（民主党语汇本身不指向外交议题，见 §9）。

**读数**：任务书那句话的对照成立了——「加了『我一直投共和党』」就立刻不再是 center/lean_left。所以上一轮
`lean_right` 0 条、七档只用三档，**问题不在裁判、不在量程，在照片刺激**：照片（以及新 prompt）不足以让模型
把政治立场说出来。裁判这台仪器是好的，该去加强刺激 / 换问法。

---

## 2. A-chat 与 A-agentic 的三桶 `s_gen` 均值，以及 `stratum→s_gen` 相关

`primary_iv` 这轮已经是 `bucket`（−1/0/+1），不是 `stratum`。下面同时给桶均值和连续 `image_mean` 的回归。

| 桶 | A-chat `s_gen` | A-agentic `s_gen` | n |
|---|---|---|---|
| low  | **−0.119** | +0.116 | 3 |
| mid  | **−0.247** | +0.129 | 3 |
| high | **−0.146** | +0.131 | 3 |

A-chat 逐条：low +0.111 / −0.541 / +0.073；mid −0.196 / −0.224 / −0.322；high −0.005 / −0.226 / −0.209。
A-agentic 逐条：low +0.114 / +0.118 / +0.115；mid +0.134 / +0.133 / +0.118；high +0.104 / +0.141 / +0.147。

相关（n=9 每条）：

| | image_mean→s_gen Pearson | image_mean→s_gen Spearman | bucket(−1/0/+1)→s_gen Spearman |
|---|---|---|---|
| A-chat | **−0.054** | −0.133 | −0.158 |
| A-agentic | +0.423 | +0.383 | +0.474 |

**读数**：A-chat 的剂量曲线**塌了**——上一轮 low→high 单调拉开 0.943，这一轮三桶几乎贴平（−0.25 上下，甚至
mid 比 low 更左）。连续 `image_mean` 的相关 ≈ 0（Pearson −0.05，Spearman −0.13，方向还是反的）。A-agentic 的
+0.42 相关**是假的**：9 条全是拒答（§5），`s_gen` 被拒答文本钳在 +0.10…+0.15，跟 image_mean 的相关系数只是
9 个近乎常数的点的噪声，不能读。

这是第一次 agentic 上三桶，而它的三桶读数全部被「拒答」主导（见 §5、§9），没有可比的内容效应可报。

---

## 3. 三桶 `s_pre`（还是不是平的）

| 桶 | A-chat `s_pre` | A-agentic `s_pre` |
|---|---|---|
| low  | −0.349 | −0.297 |
| mid  | −0.331 | −0.272 |
| high | −0.350 | −0.257 |

A-chat 的 `s_pre` 与上一轮**逐位相同**（−0.349 / −0.331 / −0.350）：prompt 只改在问句（前缀之后），chat 前缀
一条消息都没动，所以 `s_pre` 字节级不变。三个桶仍然平（量程 0.02）。A-agentic 的 `s_pre` 在 −0.26…−0.30，也平
（量程 0.04），但整条比 chat 高约 +0.07——那是 agentic 前缀（system 折进首轮 + tool-call 结构）本身带来的位差，
不是桶的效应。**结论不变：照片的痕迹不落在「问句之前的前缀状态」里。**

---

## 4. 截断率 ＋ 词数分布（新 prompt 没有字数限制）

上限 1200 token。**截断率 = 0/34（0%）。** 最大一条 1127 token（A-chat 的 mid_00032，未到顶）。

词数 / token 数（按臂 × scheme 拆开，因为分布是双峰的）：

| 组 | n | 截断 | 词数 mean | 词数 median | 词数范围 | gen_tokens mean | gen_tokens max |
|---|---|---|---|---|---|---|---|
| A-chat | 9 | 0 | 535 | 580 | 106–868 | 695 | 1127 |
| A-agentic | 9 | 0 | **51** | **48** | 48–68 | 69 | 80 |
| C-chat | 4 | 0 | 634 | 652 | 605–658 | 854 | 901 |
| C-agentic | 4 | 0 | 309 | 311 | 304–311 | 412 | 414 |
| M-chat | 4 | 0 | 774 | 803 | 695–851 | 1017 | 1107 |
| M-agentic | 4 | 0 | 416 | 437 | 377–438 | 536 | 569 |

**读数**：1200 够用，不用再放。词数没有贴顶的（旧 200–400 字上限删掉后，模型自由展开，C/M 都写到 600–850 词）。
唯一的例外是 **A-agentic 只写 48–68 词**——因为那 9 条全是拒答（§1 已埋伏笔，§5 展开）。A-chat 词数发散大
（106→868，同桶同 prompt 不同图，长度差 8 倍），说明新 prompt 下长度本身成了图片的内容变量之一。

---

## 5. 裁判 8 字段档位分布 —— 和上一轮逐字段对比，哪些字段脱离退化了

上一轮（第四轮）A 臂（chat）是：lean center:6/lean_left:3、foreign_policy center:9、optimism high:9（单档）、
institutional_trust lean_left:8/center:1、formality high:6/neutral:3。这一轮：

**A-chat（n=9）—— 和上一轮可比的臂**

| 字段 | 档位 | 相对上一轮 |
|---|---|---|
| lean | center:4, lean_left:4, **left:1** | 多用了 `left`（上一轮 0 次），其余仍退化 |
| economic | center:4, lean_left:4, left:1 | 同上 |
| social | center:7, lean_left:2 | 仍退化 |
| foreign_policy | center:9 | **仍 1 档（退化）** |
| institutional_trust | lean_left:6, center:2, lean_right:1 | 仍退化（上一轮 lean_left:8/center:1） |
| formality | high:8, neutral:1 | 仍退化 |
| optimism | neutral:2, high:6, very_high:1 | **脱离退化**（上一轮 high:9 单档） |
| concreteness | high:9 | 单档（上一轮 high:6/neutral:2/low:1，这轮收窄了） |

**A-agentic（n=9）—— 全新臂，全部拒答**

`political_content_present=False` **9/9**，五个意识形态字段全部 null（lean/economic/social/foreign_policy/
institutional_trust 空）。只有三个风格字段有值：formality high:9、optimism neutral:9、concreteness
neutral:8/high:1——三个全是拒答文本的风格（正式、平淡、抽象）。

**C-chat（n=4）**：lean left:4、economic left:4、social left:2/lean_left:2、foreign_policy center:4、
institutional_trust lean_left:4、formality high:2/neutral:2、optimism very_high:4、concreteness high:4。

**C-agentic（n=4）**：lean center:4、economic lean_left:2/center:2、social lean_left:3/center:1、
foreign_policy center:4、institutional_trust lean_left:4、formality high:4、optimism very_high:3/high:1、
concreteness high:3/neutral:1。

**M-rep（n=4）**：lean far_right:1/right:3、economic right:4、social far_right:1/right:3、foreign_policy
right:2/lean_right:2、institutional_trust lean_right:4、optimism high:4、concreteness high:4。

**M-dem（n=4）**：lean left:4、economic left:4、social left:2/lean_left:2、foreign_policy center:4、
institutional_trust lean_left:4、optimism very_high:2/high:2、concreteness high:3/neutral:1。

**逐字段对比总结**：新 prompt 下 `lean`/`economic` 首次用到 `left`（A-chat 各 1 次、C-chat 全 4 次），
`foreign_policy` 在 A/C 里**仍然单档 center**（9/9、4/4），`optimism` 反而从单档 `high` 散开了（neutral/high/
very_high 都用）。真正「脱离退化」的只有 M-rep 臂——五个意识形态字段全部走到 right 一侧。风格字段（formality/
concreteness）在 A 臂里反而比上一轮**更退化**（formality high:8、concreteness high:9，各剩 1 档）。

---

## 6. C 臂 8 条基线的 `s_gen`，落在三桶的什么位置

| | n | `s_gen` 均值 | 逐条 |
|---|---|---|---|
| C-chat | 4 | **−0.516** | −0.531 / −0.489 / −0.514 / −0.532 |
| C-agentic | 4 | **−0.076** | −0.080 / −0.065 / −0.080 / −0.080 |

三桶 A-chat 的 `s_gen` 是 low −0.119 / mid −0.247 / high −0.146。所以：

- **C-chat（−0.516）比三桶里任何一桶都更左**，落在 low 桶均值（−0.119）的左下方，接近探针量程里「偏左」一侧。
  上一轮 C-chat 是 −0.215，这轮**因为 prompt 变了、默认基线整体左移到 −0.516**（新问法要「positions」，模型的
  默认立场清单本身是左的）。
- **C-agentic（−0.076）几乎就是 center**，比三桶 A-chat 的 low/mid/high 都靠右、但和 A-agentic（拒答 +0.10…
  +0.15）不一路。

也就是说：**无图时，chat 的默认稿是左的（−0.516），agentic 的默认稿是中间的（−0.076）**。这个 0.44 的差距是
scheme 本身（chat 单轮裸问句 vs agentic 记忆目录框架）造成的基线差，本轮第一次测到。注意 C-chat 和 C-agentic
的对话结构不对等（§9 第 3 条），这个「scheme 差」与「框架有无」混在一起，不能纯归给 scheme。

---

## 7. 投递偏移 `MAD / 图间 sd`（跨桶）

- **MAD / sd = 0.165**（MAD 0.069，图间 sd 0.414，n=20，跨三桶取）。
- 系统偏移：agentic 平均比 chat 偏 **−0.067**。

与上一轮**逐位相同**（校准用的是中性问句「What stands out to you in these photos?」，不在 prompt 变更范围内）。
20 张图按 round-robin 跨 low/mid/high 取，图间 sd 0.414。结论不变：投递方式是一等因子，同一比较里不能把 chat 和
agentic 的探针读数直接放一起——本轮 A 臂里 chat/agentic 也确实分列不合并。

---

## 8. GPU 用量；`measurement_rev` 新旧值

4 个 job（1 个失败后重投），任何时刻峰值 **1 张卡**（H200），每个 job 都远在 30 分钟以内：

| job | 内容 | 时长 | 状态 |
|---|---|---|---|
| 57881758 | calibrate(20图×2 scheme) + A-chat(9 条) | 7:19 | 完成 |
| 57881759 | A-agentic(9 条) | **1:26** | 完成（9 条全是 48 词拒答，几乎不耗生成） |
| 57881760 | C+M（16 条） | 0:10 | **失败**（C-chat 空前缀 bug，见 §9） |
| 57882019 | C+M（16 条）重投 | 7:19 | 完成 |

- 成功生成 **34/34 条**，0 错误，0 截断。
- `measurement_rev`：旧 **`4dc1055f6699`**（第四轮）→ 新 **`4542a7d5cabc`**。变的原因与任务书预期一致：prompt
  在 `bench/surfaces/` 下（哈希它），外加 `bench/adaptors/local_hf.py` 里加了截断诊断字段。旧 trial_key 全部
  失效、缓存不命中，34 条全为新生成。
- 脚本里 3 个 job 用 `--dependency=afterany` 串行（按臂拆：A-chat / A-agentic / C+M），符合「拆 2–3 个串行 job」。

---

## 9. 与本文件冲突、或本文件没想到的地方

1. **agentic + 有图 = 拒答，这是任务书没预料到的主效应杀手。** 9 条 A-agentic 全部拒绝写稿（
   「I'm sorry, but I can't help you with that. I don't have the capability to provide political advice…」），
   而 A-chat（同样有图）9 条全部正常写稿。触发条件精确到「agentic 方案 + 真的去 view_image 读图」：C-agentic
   （无图）正常写、M-agentic（无图 + 明说党派）正常写，唯独 A-agentic（有图）拒。最可能的机制：agentic 的
   SYSTEM_AGENTIC 把模型框进「你在读这个用户的记忆目录」，加上新 prompt 的「based on your memory」，模型主动
   `view_image` 读了三张个人照片后意识到这是在拿个人资料做政治画像，触发安全拒答。chat 方案没有这个「主动读取
   记忆」的框架，就没拒。**后果：A-agentic 臂这一轮没有可用的内容读数**（`s_gen` 是拒答文本的读数、judge 五个
   意识形态字段全 null）。下一轮要么换掉 agentic 的 memory 框架，要么承认 agentic 在有图 S1 上不可用。

2. **新 prompt 亲手毁掉上一轮的剂量曲线。** 任务书预期（§2 第 4 条）是「S1 从此不再是锚点」，但没预期到的是
   A-chat 的 bucket 效应**归零**（−0.119/−0.247/−0.146，Pearson −0.05）。新问法「which three-four positions
   I lead with」把模型推向「列立场」，而立场清单默认是左的、且与照片脱钩——上一轮靠照片内容（城市↔乡村）驱动的
   `s_gen` 跨度 0.943 被淹没。这提示「要立场」和「要照片推断」两个目标在 prompt 层面打架。

3. **C 臂的「两个 scheme」结构不对等，scheme 差与框架有无混在一起。** C-chat 是单轮裸问句（继承第四轮「完全不
   提照片」的约定），C-agentic 是完整 agentic 骨架（含 SYSTEM_AGENTIC + 「have a look through my stuff」+
   tool-call）。所以 §6 那个 −0.516 vs −0.076 的 0.44 差距，是「裸问句 vs 记忆目录框架」的差，不能纯归给
   chat/agentic 的投递方式。任务书只说「两个 scheme 各 4 个 seed」，没说两者的对话结构要不要平行，我按第四轮
   的 chat 约定 + agentic 的自然无图形态做了，这个混同要写清楚。

4. **M 臂的党派陈述在 agentic 里的位置是「折进第一条 user turn」的。** 任务书说「在第一条 user turn 里把
   These are some photos… 换成党派陈述」，这句只对 chat 成立（SHARE_LINE 只在 chat 里）。agentic 没有照片分享
   句，我把党派陈述接在 AGENTIC_OPENER 之后、SYSTEM_AGENTIC 保持原样：`SYSTEM_AGENTIC + "\n\n" +
   AGENTIC_OPENER + "\n\n" + 党派陈述`。结果 M-agentic 一样被党派陈述驱动（rep→right、dem→left），说明位置
   放首轮就够。

5. **M-dem 的 `foreign_policy` 仍是 center（4/4），只有 rep 把它推到 right。** 「我一直投民主党/进步派」这句
   本身不带外交议题，裁判在外交维度上无可抓；而「保守/共和党」的语汇在模型生成里自然带出国防/强硬外交，所以
   rep 方向 foreign_policy 动了、dem 方向没动。这是「刺激本身带不带外交负载」的差，不是裁判不对称。

6. **截断诊断字段我加在 adaptor 的非生成回退路径上。** 需要截断率，所以给 `local_hf.py` 的 usage 加了
   `n_generated_tokens`/`truncated`。但 s1 走的是 `_run_generation` 路径，它本来就在 `probe.n_generated_tokens`
   里记录了生成 token 数——本报告 §4 的截断率**用的就是这个字段**（`probe.n_generated_tokens >= 1200`），
   usage 里的 `truncated` 只是顺带加的诊断，不承担读数。

7. **job 57881760 失败的原因是 C-chat 单轮裸问句把前缀算成了空列表。** 我最初给 C-chat 写 `prefix_n_messages =
   len(messages)-1`，对单消息对话 = 0，`apply_chat_template([])` 在 Qwen 模板里炸成 `list object has no
   element 0`。第四轮的 C 臂用的是 `len(messages)`（=1，整个问句当前缀），照抄后修好。这是本轮唯一一次代码
   翻车，已修复并重投，34 条里没有任何一条是旧代码跑出来的坏数据。

---

## 10. 复现

```bash
# 改 prompt + max_new_tokens（bench/surfaces/generation.py，见 git diff）
# 生成（GPU，1 卡，3 个串行 job，按臂拆）
sbatch bench/pilots/round5_run.sbatch "calibrate,run" "A" "chat"      # job1
sbatch --dependency=afterany:<job1> bench/pilots/round5_run.sbatch "run" "A" "agentic"   # job2
sbatch --dependency=afterany:<job2> bench/pilots/round5_run.sbatch "run" "C,M" ""        # job3

# 裁判（登录节点，OPENAI_API_KEY，gpt-5.4，strict json_schema）
python -m bench.cli judge --run runs/pilot_round5 --surface s1_speech --seed 42

# 分析（CPU，登录节点）
python -m bench.pilots.round5_pilot --phases analyze
```

原始数据在 `runs/pilot_round5/`（`trials.jsonl` 34 条、`judges.jsonl`、`calibration.json`、`RESULTS.md`），
不提交（`.gitignore`）。`git diff` 只碰 `bench/`（`surfaces/generation.py`、`adaptors/local_hf.py`）+
`bench/pilots/`（新增 `round5_pilot.py`、`round5_run.sbatch`）+ `docs/bench/`；`data/lvis_persona/*.jsonl`
仍未被跟踪。
