# Benchmark 设施 ⑨ —— 六个 surface ＋ 裁判层 ＋ 第三轮 pilot

> 第三轮。实现 `bench/surfaces/generation.py`（一个 `GenerationSurface` ＋ 六份配置）、
> `bench/judges/`（`JudgeSpec` ＋ OpenAI 兼容调用器 ＋ sqlite 缓存）、两套 scheme 作为
> variant、四层测量（`s_pre` / `s_gen` / 确定性抽取 / 裁判），并跑了一个有界 pilot。
> 设计权威来源是两块 board（`board-tasks.html`、`board-judge.html`）。

---

## 1. 这轮唯一真正要的数 —— `stratum → s_pre` 与 `stratum → s_gen`

按 board 的要求，先直说结论，不修饰：**两个读数给出了相反的答案。**

### `stratum → s_pre`：平的

`s_pre` 是探针在**共享前缀末尾**（问题之前）的读数，board 说它是「照片有没有留下痕迹的干净读数」。
实测（Qwen3-VL-8B-Instruct，`combined_ideology_headwise_linear`，k=16，每个 scheme×stratum 各 8 条）：

| scheme | stratum 0 | stratum 5 | stratum 9 |
|---|---|---|---|
| chat    | −0.328 | −0.326 | −0.352 |
| agentic | −0.310 | −0.288 | −0.275 |

整条量程只有 0.02–0.04，**没有任何随 stratum 单调的趋势**。Spearman 名义上 agentic +0.65、
chat −0.24，但这是在一条平的信号上算的，没有意义。**照片在「问题之前的那个状态」上没有留下
可测的痕迹。**

### `stratum → s_gen`：不是平的

`s_gen` 是探针对**模型生成的每一个 token**取均值。同一批格子：

| surface × scheme | Pearson | Spearman | n |
|---|---|---|---|
| s1_speech · chat    | **+0.906** | +0.917 | 12 |
| s1_speech · agentic | **+0.836** | +0.828 | 12 |
| s4_bonus · chat     | +0.470 | +0.414 | 12 |
| s4_bonus · agentic  | +0.612 | +0.562 | 12 |

`stratum` 越右，生成文本的探针读数越右。s1（竞选演讲，唯一有阶段④外部标定点的任务）在两个
scheme 下都是强正相关（0.84–0.92）；s4（奖金怎么分，最隐性）也是正相关但较弱（0.41–0.61）。

**这一对数字是这轮真正的发现：照片的痕迹不是留在「生成前的前缀状态」里（s_pre 平），
而是**在生成过程中**才显现出来（s_gen 随 stratum 上升）。** 中介链「图像 → s_pre → 输出」的
中间那一段（s_pre）是断的；图像的影响是生成时通过注意力现取现用的，而不是先写进一个持久状态
再读出来。board 把 s_pre 当作「中介链的中间项」，这一点被证伪了。

### 有图/无图基线差（condition C vs E，s_gen 均值）

| surface × scheme | C（有图） | E（无图） | 差 |
|---|---|---|---|
| s1_speech · chat    | −0.070 | −0.288 | **+0.218** |
| s1_speech · agentic | −0.238 | −0.411 | **+0.173** |
| s4_bonus · chat     | +0.035 | −0.003 | +0.039 |
| s4_bonus · agentic  | +0.087 | +0.217 | −0.129 |

s1 的两个 scheme 都有清晰的「有图更偏右」基线差（+0.17~+0.22），s4 则没有一致方向 ——
这与 `s_gen` 相关的强弱（s1 强、s4 弱）一致。

---

## 2. 投递偏移校准 —— `MAD / 图间 sd`

20 张图、40 次前向（chat/agentic 各一次），k=16：

- **MAD / sd = 0.288**（MAD 0.069，图间 sd 0.240，n=20）
- 系统偏移：agentic 平均比 chat 偏 **−0.055**（20/20 张图方向一致）。

上一轮 k=16 是 **0.22**，这轮 **0.29**，略高。差异来自两处：这轮 agentic scheme 的转录更长
（system＋opener＋`list_dir`＋三张 `view_image`＋ack），以及图集换成了 explore_v2 的跨层样本。
方向与上一轮一致（agentic 更负），量级同一档。**结论不变：投递方式是一等因子，同一比较里不能
把 chat 和 agentic 的 `s_img`/`s_pre` 直接放一起** —— 这轮的 `s_pre` 按 scheme 分着报就是为此。

> 注：第一版校准误用了 explore_v2 里**前 20 张**（全是 stratum 0 的左倾图），图间 sd 被压到
> 0.074，得到 MAD/sd=1.23 的假值。改成跨层取图后才是上面这个 0.29。这条教训记在这里：
> **校准的图集必须和实验一样跨全部分层**，否则比值没有意义。

---

## 3. 裁判：档位分布 —— 全部退化

裁判跑在 52 条生成上（s1 × s4，gpt-5.4，strict `json_schema` + `logprobs`）。按 board 第二条
判据（某一档 > 60% 或只用上 ≤3 档 = 退化，逐字段判）：

**s1_speech（n=26）**

| 字段 | 档位 | 退化 |
|---|---|---|
| lean | center 13 / lean_left 12 / lean_right 1 | 是（3 档，且**从不偏右**）|
| economic | center 13 / lean_left 11 / lean_right 2 | 是 |
| social | center 13 / lean_left 12 / lean_right 1 | 是 |
| foreign_policy | center 26 | 是（1 档）|
| institutional_trust | lean_left 20 / center 4 / lean_right 2 | 是（77% 单档）|
| formality | neutral 18 / high 8 | 是 |
| optimism | high 26 | 是（1 档）|
| concreteness | high 20 / neutral 6 | 是 |

**s4_bonus（n=26）**

| 字段 | 档位 | 退化 |
|---|---|---|
| equality_vs_merit | center 8 / lean_right 7（另 11 条 `political_content_present=false` → null）| 是 |
| hedging | a_great_deal 20 / quite_a_bit 5 / moderately 1 | 是（77% 单档）|
| formality | high 24 / neutral 1 | 是 |
| optimism | high 23 / very_high 2 | 是 |
| concreteness | high 25 | 是（1 档）|

**八个字段全部退化，包括三个本该当「特异性对照」的 5 点控制（formality/optimism/concreteness）。**
主刻度七点量表实际只用到 center / lean_left / lean_right 三档。这条退化不是测量 bug，而是
**模型文本本身的事实**：Qwen3-VL-8B 看着这些图写出来的竞选稿和奖金意见，在文本层面被裁判
一致判成「中间/略左」，而且乐观、正式、具体几乎全踩在「high」上 —— 换句话说，**模型在文本层
面上把自己的政治表达钳在中间偏左，从不写右倾内容**（26 条 s1 里 `lean_right` 只有 1 条，
`far_right` 0 条）。而探针 `s_gen` 却能读出随 stratum 的右移（见 §1）。

这构成这轮第二个真正发现：**探针（内部状态）和裁判（文本）看到的东西不一样。** 照片把模型的
内部状态推向右，但模型**不把它说出来**；说出来的是去政治化、中间偏左的「安全文本」。board 的
七点量表是在「文本会跟着照片走」的假设下设计的，这个假设在小模型上不成立。board 预判了「退化
先改锚点措辞、再退到 logprob 期望」—— 但这里退化是逐字段且全部发生，改锚点措辞救不了（因为
不是措辞问题，是模型输出本来就窄），真正的出路是像 s3/s5/s6 那样**把主指标改成确定性抽取或探针**，
裁判降级为方向辅助。

---

## 4. 裁判 endpoint 三条能力查清结果

登录节点上查过 `$DEEPSEEK_API_KEY`、`$OPENAI_API_KEY` 和 opencode 配置目录。**只报有无和变量名，不打印内容：**

1. **有没有 key**：有。`$OPENAI_API_KEY`（指向 api.openai.com，可用 gpt-5.4 / gpt-5.4-mini 等）
   和 `$DEEPSEEK_API_KEY`（指向 api.deepseek.com，可用 deepseek-v4-flash / deepseek-v4-pro）。
2. **strict `json_schema`**：OpenAI **支持**（`response_format={type:json_schema,strict:true}` 实测
   200 OK，gpt-5.4 与 gpt-5.4-mini 都通过）。DeepSeek **不支持**（实测 400
   `"This response_format type is unavailable now"`），只有 JSON mode（`{type:json_object}`）。
3. **`logprobs` / `top_logprobs`**：两家都支持。按 board「支持就打开、存下来、平常不用」，调用器
   请求 `logprobs=true, top_logprobs=5` 并存进缓存；主刻度仍是七点标签，不依赖它。

**实现后果**：调用器默认走 strict schema；endpoint 拒绝时自动退到 JSON mode ＋ 代码里做 schema
校验 ＋ 一次重试（DeepSeek 的路径就是这么铺的）。这一事实写进了 `bench/judges/caller.py` 的
docstring。这轮裁判真实跑在 OpenAI gpt-5.4 上，**不是 fixture**（`bench/judges` 的单元测试用
fake client，不依赖 key；但 pilot 的 52 条是真实 API 调用，落在 `runs/pilot_round3/judges.jsonl`）。

---

## 5. GPU 用量

- **3 个串行 job**（`--dependency=afterany`），每个 1 张 H200，任何时刻只占 1 张卡：
  - job 1（校准 + stratum 0）：**9:00**
  - job 2（stratum 5）：**5:09**
  - job 3（stratum 9）：**6:09**
- 另加 1 次校准重跑（srun，约 2 分钟，修跨层取图）。
- 峰值 **1 张卡**，单 job 最长 **9 分钟**，远在 30 分钟之内。

**一个必须写进记录的工程坑**：`device_map="auto"`（accelerate dispatch）会让 Qwen3-VL 的自回归
生成慢 ~20 倍（实测 4 tok/s vs 83 tok/s），因为它给每个模块包了一层设备转移 hook。上一轮只跑
prefill 的 choice surface 所以没暴露，这轮一跑生成就现形。改成 `from_pretrained(...).to(device)`
单卡直载后恢复 ~70–83 tok/s。`bench/adaptors/local_hf.py` 已修；但共享节点上的**另一份间歇性
争抢**仍把个别 trial 拖到 100s（同 400 token，争抢时 100s、空闲时 6s），所以三份 stratum 拆开
串行跑，而不是赌一个 30 分钟的整 job。

---

## 6. 与 board 冲突 / board 没想到的地方

1. **`s_pre` 平、`s_gen` 有信号（最重要）**。board 把 `s_pre` 定义为「照片有没有留下痕迹的干净
   读数，也是中介链的中间项」。实测 s_pre 与 stratum 无单调关系，而 `s_gen`（生成的 token）才有。
   **照片的影响是生成时现取的，不是写进前缀末尾的持久状态。** 下一轮的中介分析不该再把 s_pre
   当中介项，s_gen 才是那个随剂量走的量。

2. **裁判七点量表全部退化，且「特异性对照」也退化**。board 用 formality/optimism/concreteness
   当「不该跟着照片分层变」的对照，实际上它们全卡在 high，连变都没变 —— 对照本身没量程，就无从
   说「它没变」。真正的负结果不在这些 5 点控制里，而在「模型文本永远中间偏左、从不偏右」。

3. **`s3_digest` 的数据 gap**。board 说「十二条标题及其 Ad Fontes 分数，数据在
   `data/adfontesmedia.csv`」。实际上那个 CSV 只有 **outlet 名 ＋ bias_mean**（400 家媒体），
   **没有标题文本**。实现里用 6 左 + 6 右的 outlet 名顶替「标题」，并按 `(item, scheme)` 做确定性
   洗牌、把顺序写进 `variant`。这意味着 s3 提示词里会自然出现「Democrat」这类词 —— 这不违反
   board 的「提示词不许出现政治词」（那条针对的是**框架**，s3 的 outlet 名是**数据**，其倾向正是
   要测的量），但需要在论文里说清「这轮喂的是 outlet 名，不是真标题」。

4. **`s3` 的随机顺序要求 `bench run` 先 build 再取 key**。board 说「顺序每次随机、必须进 variant」。
   现有 harness 是用**声明好的 variant** 算 `trial_key` 的，build 出来的顺序进不去。所以改了
   `cmd_run`：先 `surface.build()`，再用 `trial.variant`（含 s3 顺序）算 key 和写记录。这是对
   上一轮 harness 的一处必要修改，choice surface 的行为不变（build 不改 variant）。

5. **基线塌缩：64 → 52**。任务书按「2×2×4=16 条基线」估 ≈64 条。但无图基线的对话是
   item-invariant 的（同一 scheme 内逐字节相同），harness 只跑一次并按 (surface, scheme) 广播，
   所以基线是 4 条生成而不是 16 条。实际 48 图 + 4 基线 = 52 条。这是 harness 本来的语义，不是
   偷工。

6. **`s4_bonus` 的 `political_content_present` 大量为 false**（26 条里 11 条 null）。board 把
   s4 描述成「零政治词、最隐性」，实测它也确实经常被模型/裁判判成「不含政治」—— 这反过来印证了
   s4 是「最可能出不来的那个」，和 board 的判断一致，只是「出不来」的方式是**政治内容判定直接为假**，
   而不是倾向分聚在中间。

---

## 7. 验收对照

1. 已有测试全绿 + 新测试：**197 通过**（原 161 + 新增 36，含 generation surface、judge 层）。
2. 不变量测试：同一 scheme 内六 surface 的 `s_pre` 完全相同 —— 单测 + 真实数据 24/24 格子一致。
3. key 测试：只改 `scheme` 变 key；只改 s3 标题顺序也变 key（单测覆盖）。
4. 每条记录都有 `probe.top_k`（k=16 写进 `measurement_rev`，k=8 另存 `probe.k8`）。
5. opencode 跑这六 surface 时明确 BLOCKED（`missing required capability: activations`），不静默降级。
6. 裁判输出过 schema 校验；档位分布报出（全部退化，见 §3）。
7. `git diff` 只碰 `bench/` + `docs/bench/`；`data/lvis_persona/*.jsonl` 的 28 个文件仍未被跟踪。

## 8. 复现

```bash
# 生成（三个串行 job，1 GPU，各自 <10 min）
STRATUM=0 sbatch bench/pilots/round3_run.sbatch
# ... 等它结束，再 STRATUM=5、STRATUM=9（或用 --dependency=afterany 链起来）

# 裁判（登录节点，走 OPENAI_API_KEY）
python -m bench.cli judge --run runs/pilot_round3 --seed 42

# 分析
python -m bench.pilots.round3_pilot --phases analyze --run runs/pilot_round3
```

原始数据在 `runs/pilot_round3/`（`trials.jsonl`、`judges.jsonl`、`calibration.json`、`RESULTS.md`），
不提交（在 `.gitignore` 里）。
