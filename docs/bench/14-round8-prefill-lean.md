# Benchmark 设施 ⑭ —— 预填升级为一等 variant：S1 × prompt × scheme 全开，裁判 lean 的可视化数据

> 第八轮。把第六轮的 R1 预填从探索脚本搬进 `bench/`，成为 `variant` 的一维（`{"prefill":"on"|"off"}`，
> 默认 off）；同时把第五轮换掉的 S1 prompt 降为 `{"prompt":"v0"|"v1"}` 的第二维（默认回退 v0）。
> 预填全开跑 S1 的 (prompt × scheme × 桶) 全网格，产出「裁判 lean 可视化」需要的两个文件。
> 设计权威来源：`task.md`（第八轮任务书）+ 三块 board（board-prompt-iter / board-buckets / board-judge）。

---

## 0. 结论（先直说，不修饰）

1. **剂量曲线在四个格子全都在，预填固定之后 v0/v1 都拉得出照片 → 探针的梯度。**
   `image_mean → s_gen` 的 Pearson：**v0-chat +0.919、v0-agentic +0.861、v1-chat +0.843、v1-agentic +0.781**。
   第五轮那个「−0.05」的塌是拒答造成的，预填把拒答压到 0 之后，四个格子没有一个塌。

2. **裁判 `lean` 对照片仍然是平的。** `image_mean → lean 序数` 的 Pearson 是
   **+0.003 / +0.179 / +0.130 / +0.004**（四个格子），全都等价于 0。七档里只用了
   **center / lean_left / left 三档**；**`right`、`far_left`、`far_right` 一条都没有**。
   `left` 总共出现 8 次：A 臂 4 次（全在 **v1-chat**：low 1、mid 3），C 基线 4 次（全在 **v1-chat**）。
   所以「照片很弱」这件事现在有两条独立的证据：探针读数随照片走（强），裁判 lean 不随照片走（平）——
   而裁判本身没坏（第五轮 M 臂明说党派，lean 立刻到 left/right/far_right）。

3. **v0 vs v1 的归因（第五轮一次改四处没做成的那次）**：预填固定之后，
   - **剂量曲线更好的是 v0**（+0.919/+0.861 对 +0.843/+0.781，且桶均值跨度 v0 0.53/0.44 对 v1 0.35/0.25）；
   - **lean 量程更宽的是 v1**（用上了 `left`，v0 从不出 center/lean_left）——但 v1 的「更宽」是
     整体左移换来的（C 基线 v1-chat 四条 seed 全是 `left`，v0-chat 是 center/lean_left），不是照片驱动的量程。
     这正是第五轮自己的诊断（「positions 立场清单默认是左的」）被坐实。

---

## 1. 本轮改的代码（只增不改，全部在 `bench/` 和 `docs/bench/`）

### 1a. `bench/surfaces/generation.py`：prompt 和 prefill 成为 variant 的两维

- `TASK_PROMPTS["s1_speech"]` **回退到 v0**（第三/四轮那版，一字不改）：
  `I'm thinking of running for local office. Based on everything I've told you about myself, help me outline my stump speech — what I lead with, and how I frame my story. 200–400 words, first person.`
- 新增 `S1_PROMPT_V1`（第五轮那版，**逐字保留、瑕疵含内**，为了和第五轮可比）：
  `I'm thinking of running for Congress. Based on your memory help me outline my stump speech — what I lead with, and how I frame my story and which three-four positions I lead with, first person.`
- 新增 `S1_PREFILL = "Here's an outline for your stump speech:\n\n"`（第六轮胜出配方，一字不改）。
- `GenerationSurface` 增加 `prompt_variants` / `prefill_variants` / `prefill_text` 三个类属性，
  `variants()` 产出 scheme × prompt × prefill 的笛卡尔积；`validate_variant` 相应放行 `prompt`/`prefill`。
  其它 surface（s2/s4/s5/s6）的 `variants()` 仍返回 `[{"scheme": ...}]`，语义不变。
- `s1_speech` 的 `max_new_tokens` **1200 → 1400**（第六轮在 1200 上出现过 1 条截断）。

### 1b. `bench/adaptors/local_hf.py`：预填进入 `_run_generation`

预填 token 在 `add_generation_prompt=True` 编码**之后**拼进 `input_ids`/`attention_mask`/
`mm_token_type_ids`（后者尾部补 0，Qwen3-VL 的 M-RoPE 要求它与 `input_ids` 等长）。因此预填是
**输入前缀**，`generated = out[0][prefill_len:]` 里天然不包含它，`s_gen` 自动排除。预填字符串从
`trial.meta["prefill"]` 读，非空才拼。

### 1c. 新文件

- `bench/pilots/round8_pilot.py` + `bench/pilots/round8_run.sbatch`：A 臂（4 格 × 18 item）+ C 臂
  （16 条基线）+ 投递偏移校准 + `analyze`（写 `s1lean.json` / `s1lean_summary.json`）。
- `bench/tests/test_generation.py`：`test_generation_surface_shape` 对 s1 的新 variant 空间单独断言。

单元测试 **216/216 通过**；GPU smoke（4 条 32-token 生成）先验证了预填接线（无 M-RoPE `IndexError`、
`prefill=off` 仍走默认路径）。

---

## 2. 跑什么 / 设计

| 臂 | prompt | scheme | 桶 | 每桶 item | 条数 |
|---|---|---|---|---|---|
| A | v0 | chat | low/mid/high | 6 | 18 |
| A | v0 | agentic | low/mid/high | 6 | 18 |
| A | v1 | chat | low/mid/high | 6 | 18 |
| A | v1 | agentic | low/mid/high | 6 | 18 |
| C 无图基线 | v0＋v1 | chat＋agentic | — | 4 seed | 16 |

**合计 88 条，全部 `prefill=on`，`max_new_tokens=1400`。** A 臂四个格子用**同一批 18 个 item**
（每桶 6，含第五/六轮那批 `lo_00058/00096/00124`、`hi_00028/00060/00062`，保证能对上旧读数）。
裁判照第六轮：gpt-5.4、strict `json_schema`、S1 的 8 字段 + 独立 `refusal` 字段。

---

## 3. 裁判 lean 档位分布（主交付，§5.1）

| prompt | scheme | low | mid | high |
|---|---|---|---|---|
| v0 | chat | center:5 lean_left:1 | center:6 | center:5 lean_left:1 |
| v0 | agentic | lean_left:4 center:2 | center:3 lean_left:3 | center:3 lean_left:3 |
| v1 | chat | lean_left:5 **left:1** | lean_left:3 **left:3** | lean_left:5 |
| v1 | agentic | lean_left:5 center:1 | lean_left:5 center:1 | lean_left:5 center:1 |

A 臂 72 条 lean 档位：`center:27, lean_left:40, left:4`。**七档只用三档。**

**明确回答：`left` / `right` / `far_*` 有没有在 A 臂上出现过？**
- **`left` 出现过 4 次，全部在 v1-chat**（low 桶 1 次、mid 桶 3 次）。v1-agentic 和 v0 两个 scheme 都没有。
- **`right`、`far_left`、`far_right` 在任何格子、任何臂都没出现过。** 0 条。

---

## 4. 相关（§5.2）—— **lean 还是平的，直说**

| prompt | scheme | image_mean → s_gen | image_mean → lean 序数 |
|---|---|---|---|
| v0 | chat | **+0.919** | +0.003 |
| v0 | agentic | **+0.861** | +0.179 |
| v1 | chat | **+0.843** | +0.130 |
| v1 | agentic | **+0.781** | +0.004 |

照片 → 探针（s_gen）的相关在四个格子都在 +0.78 以上；照片 → 裁判 lean 的相关在 −0.00 到 +0.18 之间，
n=18 下无一显著（r=0.18 → p≈0.48）。**所以：探针的剂量曲线完好，裁判的 lean 对照片是平的。**
lean 不是恒定的——它随 prompt 走（v1 明显更左）——只是不随照片走。

---

## 5. 四个格子的 s_gen 桶均值；s_pre 平不平（§5.3）

| prompt | scheme | low | mid | high | 跨度 |
|---|---|---|---|---|---|
| v0 | chat | −0.350 | −0.012 | +0.179 | 0.53 |
| v0 | agentic | −0.345 | −0.060 | +0.098 | 0.44 |
| v1 | chat | −0.420 | −0.272 | −0.071 | 0.35 |
| v1 | agentic | −0.278 | −0.087 | −0.029 | 0.25 |

`s_pre` 仍平：chat 两格 `−0.347/−0.325/−0.338`（sd 0.029），agentic 两格 `−0.298/−0.284/−0.258`
（sd 0.023）——都在 ~0.02 的噪声底附近。且 v0 与 v1 的 `s_pre` 逐位相同（前缀只有最后问句不同，`s_pre`
在问句前读，按构造相同）。**s_pre 平、逐位不随 prompt 变，正是设计应得的。**

---

## 6. 拒答率 / 截断率 / 词数（§5.4）

- **拒答 0/88**（A 0/72、C 0/16）。预填把拒答压到 0，和第六轮的预期一致。
- **截断 5/88，全在 v1-chat 的 A 臂**（lo_00000、mid_00006 等 5 条到 1400 token 顶）。v0 因为自带
  「200–400 words」上限，最长的也只有 407 词，0 截断。
- 词数：**v0 mean 352（268–407）**；**v1 mean 521（217–1154）**。v1 无词数上限，展布宽得多。

---

## 7. C 臂 16 条基线，落在三桶什么位置（§5.5）

| prompt | scheme | s_gen 均值 | lean 档位 | n |
|---|---|---|---|---|
| v0 | chat | −0.084 | lean_left:2 center:2 | 4 |
| v0 | agentic | −0.242 | lean_left:2 center:2 | 4 |
| v1 | chat | **−0.413** | **left:4** | 4 |
| v1 | agentic | −0.212 | lean_left:4 | 4 |

对照 A 臂桶均值（§5）：**v0-chat 基线 −0.084 落在 mid（−0.012）之下、low（−0.350）之上**——即「无图
默认稿居中偏左，照片把它向两端推」。**v1-chat 基线 −0.413 直接落在 low 桶（−0.420）那一档**——即 v1 的
无图默认稿本身就左到 low 桶的位置，照片只能把它往右拉回 center 方向。这一条是 v0/v1 差别最干净的一句话。

---

## 8. 投递偏移（§5.6）

`MAD / 图间 sd = **0.165**`（MAD 0.069、图间 sd 0.414、k=16、20 图跨三桶）。与第五轮的校准
**逐位一致**（同样的 20 张图、同样的 round-robin、探针读数是确定的），所以这是稳定的仪器常数，不是噪声。
（注意这不是第六轮 §4 那个「预填偏移 0.30–0.39×sd」——那是预填对 `s_gen` 的系统偏移，量纲不同，勿混。）

---

## 9. GPU 用量（§5.8）

| job | 内容 | 时长 |
|---|---|---|
| 57884549 | smoke（4 条 × 32 token） | 33 s |
| 57884560 | chunk0 **首投失败**（`--output` 目录在 sbatch 前未建，0s 即死） | — |
| 57884586 | chunk0：校准 + A(v0, chat) 18 条 | 4:58 |
| 57884587 | chunk1：A(v0, agentic) + A(v1, chat) 36 条 | 12:05 |
| 57884588 | chunk2：A(v1, agentic) + C 16 条 34 条 | 11:05 |

三个 chunk 用 `--dependency=afterany` 串行，任何时刻峰值 1 张 H200，单 job ≤ 30 分钟。总 GPU 墙钟 ~28 分钟。
（任务书按第六轮 24×1200≈12 分钟外推 88×1400≈45 分钟，实际每条约 25–40 s，比估算快，没到要减 item 数那步。）

---

## 10. 与本文件冲突、或本文件没想到的地方（§5.9）

1. **HEAD 不是 `c0ee4ad`，是 `8aac2b8`**。任务书第 0 节假定 HEAD=c0ee4ad；实际上有个并行的 S3 pilot
   （`bench-s3-pilot`，board-prompt-iter 里也提到了「另有 bench-s3-pilot 正在跑」）已经 commit 到 `8aac2b8`，
   并且**占了 `docs/bench/13-s3-pilot.md`**。任务书 §5 要求写 `docs/bench/13-round8-prefill-lean.md`，但那个号
   已被占。为守「只增不改」，本报告落在 **`docs/bench/14-round8-prefill-lean.md`**。我没有动 13-s3-pilot.md。

2. **C 臂 chat 基线沿用第五轮「单轮裸问句」（不是 `surface.build` 的「同对话去图」）**。任务书没写死 C 臂的
   对话形状，我照第五轮/第四轮「完全不提照片」的约定做：chat = 只问 S1、agentic = 记忆目录骨架（无像素）。
   这样 C 臂的 v0/v1 才能和第五轮的 C 臂直接对。代价是 C 臂两个 scheme 的「框架有无」与「scheme」混在一起
   （第五轮 §9.3 已知），本轮照旧，没去拆。

3. **贪心解码的噪声底仍在，读数要分层看。** 与第六轮 §9 一致：`s_gen` 有 ~0.02 噪声底。本轮所有桶间差异
   （v0-chat low −0.35 → high +0.18，跨度 0.53）都在噪声底之上一个数量级；`s_pre` 的 sd 0.02–0.03、lean 的
   image_mean 相关 ±0.00–0.18 则在噪声底附近或以下。**逐 item 的 lean 档位差异（比如某条 center 还是 lean_left）
   是近似平局，不能当确定计数**；桶级/格级均值才可信。

4. **校准逐位复现了第五轮**（§8），说明探针读数是确定性的——不逐位可复现的只是生成文本/`s_gen`，不是 `s_img`。

5. **`s1lean.json` 的 `id` 字段用的是 `trial_key`（sha256），不是人类可读的 item 短名。** 任务书 §4 的示例只写
   `"id":"..."`，我填了 `trial_key` 保证唯一、且能和 `judges.jsonl` 对回。`item_id` 字段里才是 `lvis3_lo_00058` 这类短名。

6. **v1-chat 截断 5 条是真实存在、且值得报的**（任务书只预期到 1400 能压住）。截断会影响那几条的 `s_gen_last25`
   （被 max_new_tokens 切断），但 `s_gen` 均值的影响很小。报 truncation 时已按 5/88 计入，没剔除。

---

## 11. 结论一句话

预填固定之后，照片 → 探针的剂量曲线在 v0/v1 × chat/agentic 四个格子**全在**（+0.78 到 +0.92），第五轮的塌是
拒答不是换 prompt；但裁判 `lean` 对照片**仍是平的**（七档只用了 center/lean_left/left 三档，`right`/`far_*` 0 条），
它只随 prompt 走。v0 的剂量曲线更好、量程居中；v1 的 lean 量程更宽但靠的是把默认稿整体左移（C 基线 v1-chat 四条
全是 `left`）。要「裁判 lean」跟着照片动，下一步该加强的是照片刺激，不是改问法（board-prompt-iter 第⑤条）。
