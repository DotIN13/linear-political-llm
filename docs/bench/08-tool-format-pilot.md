# Benchmark 设施 ⑧ —— agentic tool 格式与 tool_result 里的图像（验证性 pilot）

下一轮想把刺激图从 user turn 挪进一段 **agentic 记忆检索**的对话：
system 授权 `/memory/user` → 用户说「先翻一下我的东西」→ assistant 发 `list_dir` →
tool 返回三个中性文件名 → assistant 发 `view_image` → **tool 返回一张图** →
assistant 说「看完了」→ 用户提问。

这份文档只回答一件事：**这个形状在本地这两个模型上跑不跑得通、模型是不是真读了那张图、
探针读数还认不认**。它不是实现，`bench/` 现有模块一行未动；所有代码在
`bench/pilots/tool_format_probe.py` + `bench/pilots/toolfmt.sbatch`，
结果在 `runs/pilot_toolfmt/`（不提交）。

跑法：Qwen3-VL-8B-Instruct 与 gemma-4-E4B-it 各一个 job，各 1 张 H200，
实际耗时 **65 s / 76 s**（`sacct`），远在 30 min 之内。

**一句话结论：原生 tool 格式可用，不需要退到文本转录。** 详见下面五问。

---

## Q1 · 模板吃 tool 角色（吃，但有两个必须知道的坑）

`processor.apply_chat_template` 对 `assistant + tool_calls` 和 `role: "tool"` 都渲染正常，
四种投递方式（M1/M2/M3/M1b）**render 和 encode 全部 ok**。Qwen 渲染出来的 tool 段落长这样：

```
<|im_start|>assistant

<tool_call>
{"name": "view_image", "arguments": {"path": "/memory/user/img_0417.jpg"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
<|vision_start|><|image_pad|><|vision_end|>img_0417.jpg
</tool_response><|im_end|>
<|im_start|>assistant
I've looked through your files.<|im_end|>
```

**坑 1：Qwen 没有 tool 这个角色 token。**`role: "tool"` 被模板包成 `<|im_start|>user` +
`<tool_response>` 文本标签。也就是说「图放在 tool_result 里」在 token 层面
**就是「图放在一个带 `<tool_response>` 包装的 user turn 里」**。
这决定了下一轮能声称的东西：agentic 与非 agentic 的差别是**文字框架**，不是**通道**。

**坑 2：每条消息的 `content` 必须是 list。**`content` 传字符串时 `tokenize=True` 直接崩：

```
File "transformers/src/transformers/processing_utils.py", line 1807, in apply_chat_template
    visuals = [content for content in message["content"] if content["type"] in ["image", "video"]]
TypeError: string indices must be integers, not 'str'
```

因为那段扫描是**不看 role 的**（这也正是 tool 里的图能被收进 `batch_images` 的原因）。
代价是发 tool_call 的 assistant 消息得写成 `content: [{"type": "text", "text": ""}]`，
而模板里 `if (loop.first and message.content)` 判定这个非空 list 为真，
于是 `<|im_start|>assistant` 后多出一个空行（上面那段可以看到）。无害，但要知道它是我们造成的。

Gemma 也吃，`role: "tool"` 直接渲染成 `<|turn>tool`：

```
<|turn>model
<|tool_call>call:view_image{path:<|"|>/memory/user/img_0417.jpg<|"|>}<tool_call|><turn|>
<|turn>tool


<|image|>

img_0417.jpg<turn|>
<|turn>model
I've looked through your files.<turn|>
```

**坑 3（只在 Gemma）：system turn 的 content 不能是 list。**Gemma 模板对首条消息做
`messages[0]['content'] | trim`，list 会被原样字符串化，prompt 里出现
`[{'type': 'text', 'text': "You are a helpful assistant..."}]`；而改成字符串又会撞上坑 2 的崩溃。
唯一两边都干净的形状是**把 system 文本折进第一条 user turn**（脚本的 `--system-in-user`）。

---

## Q2 · 图像在 tool_result 里：token 数一样，位置在 tool 块内

同一张图（`000000000034.jpg`），四种投递方式：

| mode | 总 token | `num_image_tokens` | 图像 token 起止 | pixel_values |
|---|---|---|---|---|
| M1 user (基线) | 301 | **260** | [15, 274] | [1040, 1536] |
| M2 tool | 671 | **260** | [359, 618] | [1040, 1536] |
| M3 text | 408 | **260** | [122, 381] | [1040, 1536] |
| M1b user-in-agentic-ctx | 671 | **260** | [385, 644] | [1040, 1536] |

Gemma 同样：四种方式都是 260（`<|image|>` = 258880），M1 [12, 271] / M2 [255, 514]。

- **数量对**：M1 与 M2 完全一致，20 张图逐张核过（`num_image_tokens` 取值
  {192, 240, 260, 300, 320, 360, 361, 400}，随分辨率变，M1 == M2 全部成立）。
- **位置对**：M2 的 260 个 token 是**一段连续区间**（20/20 张都只有一个 span），
  紧邻上下文是

  ```
  ...<|im_start|>user\n<tool_response>\n<|vision_start|>   [260 个 image_pad]   <|vision_end|>img_0417.jpg\n</tool_response>...
  ```

  即**落在 tool 块内部**，没有被挪到最前面。

---

## Q3 · 模型真读了（关键一问，控制组才是证据）

20 张 LVIS 图，每张一个面积 ≥ 25% 的 frequent 类别；干扰类别取自该图的
`neg_category_ids`（LVIS 已人工核实**不在图里**），且与真类别不共词、head noun 不同。
问题 `Which of these is in the photo: a {A} or a {B}? Answer with one word.`，
选项顺序 10/10 平衡，生成 12 token，**子串/head-noun 匹配判对错**，不用 logprob。

四个投递臂全部 20/20，说明「模板没崩」的下一步是必须的：三个控制臂。

**Qwen3-VL-8B-Instruct（n=20/臂）**

| 臂 | 准确率 | correct | wrong | neither | 说明 |
|---|---|---|---|---|---|
| M1 user | **1.00** | 20 | 0 | 0 | 基线 |
| **M2 tool** | **1.00** | 20 | 0 | 0 | **要验的那个** |
| M3 text | **1.00** | 20 | 0 | 0 | 退路 |
| M1b user-in-agentic-ctx | 1.00 | 20 | 0 | 0 | 附加臂：同上下文、图移回 user turn |
| C0 无图（只有问题） | 0.50 | 10 | 9 | 1 | 语言先验 = 随机 |
| C1 user 错配图 | 0.50 | 10 | 8 | 2 | 换成别的图 → 掉回随机 |
| C2 tool 错配图 | 0.60 | 12 | 6 | 2 | 同上 |

**Gemma-4-E4B-it（n=20/臂）**

| 臂 | 准确率 | correct | neither |
|---|---|---|---|
| M1 user | 0.95 | 19 | 1 |
| **M2 tool** | **0.95** | 19 | 1 |
| M3 text | 0.95 | 19 | 1 |
| M1b user-in-agentic-ctx | 0.95 | 19 | 1 |
| C0 无图 | 0.00 | 0 | 20 |
| C1 user 错配图 | 0.20 | 4 | 13 |
| C2 tool 错配图 | 0.15 | 3 | 16 |

控制臂是全部说服力所在：无图时 Qwen 只有随机水平（10/20），
把 tool 里的图换成另一条目的图，准确率立刻掉回随机（12/20，二项 p ≈ 0.5）；
Gemma 更干脆，无图时回「I need a photo to answer your question.」，
错配时 16/20 直接回 **"Neither"**。答案跟着**送进 tool_result 的那张图**走，不是跟着先验走。

错配臂的两条真实输出（Qwen / Gemma，问的都是 `zebra or boat`，送进去的是一张公交车）：

```
C2_tool_mismatch 000000000034.jpg  delivered=000000000471.jpg (bus)  out='boat'
C2_tool_mismatch 000000000034.jpg  delivered=000000000471.jpg (bus)  out='Neither'
```

**M2 没有失败，所以没有失败样本可贴**；上面这两条是「模型确实在看那张图」的反证据。

---

## Q4 · 探针读数：M1 与 M2 高度一致，但有一个小的系统偏移

`combined_ideology_headwise_linear`，20 张图，`s_img` = 图像 token 上探针分的均值。
**用的 k = 16**（`probes/headwise_linear_probe.py::score_samples` 默认 `k=16`，
`token_scoring.py --top-k` 默认 16，`bench/adaptors/local_hf.py` 默认 `top_k=16` —— 三处一致）。
k 扫描是纯后处理：一次前向捕获所有 k 用到的模块的并集，再分别打分。

| k | 选中的头数 | pearson r | spearman r | 平均绝对差 | mean(M2−M1) | sd(s_img, M1) | MAD / sd(M1) |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 0.977 | 0.968 | 0.0405 | +0.0196 | 0.269 | 0.15 |
| **16** | 16 | **0.971** | 0.958 | **0.0447** | −0.0197 | 0.207 | 0.22 |
| 32 | 32 | 0.957 | 0.971 | 0.0364 | +0.0195 | 0.129 | 0.28 |
| 64 | 64 | 0.955 | 0.950 | 0.0394 | +0.0312 | 0.093 | 0.43 |
| 96 | 96 | 0.961 | 0.941 | 0.0318 | +0.0252 | 0.078 | 0.41 |

读法：

- **相关性够高**（r ≈ 0.95–0.98，各 k 都是），图排序在两种投递下基本不变 → agentic scheme
  **可以保留探针测量**。
- **但不是同一把尺子**。M2 相对 M1 有 ~0.02–0.03 的偏移，且平均绝对差是**图间标准差的 15%–43%**。
  k 越大越糟：多头平均把图间方差压小了（sd 0.269 → 0.078），M1/M2 的偏移却没跟着缩。
  所以 **k 不是越大越好**，k=8/16 的信噪比反而最好。
- 结论到设计上：**同一个比较里不要混投递方式**。要比 agentic 与非 agentic，
  得把投递方式当成一个显式因子测，不能默认它们的 `s_img` 可以直接放在一起。
- `k` 必须写进结果（上一轮漏了）。probe metadata 里不带 k，光看权重文件复现不出读数。

---

## Q5 · Gemma：本地有两个

`/home/tzhang3/jevans/models/`（= `/project/jevans/tzhang3/models/`，同一个目录的软链）下：

- `gemma-4-31B-it`（59 GB）
- `gemma-4-E4B-it`（15 GB） ← **本次用的这个**

`token_scoring.py::select_model_loader` 走 `model_family == "gemma4"` 或路径里含 `gemma-4`/`gemma4`
这一支，用 `AutoModelForMultimodalLM` + `torch_dtype="auto"`。**没有下载任何权重。**
Q1–Q3 见上：模板吃 tool 角色（有坑 3）、图能进 tool_result 且 token 数/位置正确、
M2 = 0.95 且错配掉到 0.15。31B 没跑（15 GB 那个已经足够回答这三问，跑 31B 会顶到时间预算）。

探针只在 Qwen 上有（`results/probes/qwen3-vl-8b-instruct/`），所以 Q4 没在 Gemma 上做。

---

## 落到下一轮 schema 上的四条

1. **可以用原生 tool 格式**，M3 文本转录留作 fallback 即可，不必作为主路径。
2. **`encode_prompts` 需要一个 `tools=` 入口。**`scripts/probes/token_scoring.py::encode_prompts`
   没有这个参数，pilot 只好在 `bench/pilots/tool_format_probe.py::encode()` 里自己写一份
   （其余全部复用 token_scoring）。要么给 adaptor 加这个入口，要么把 tools 段预渲染进 system 文本。
3. **conversation schema 得能表达 `tool_calls` 和 `role: "tool"`，且强制 content 为 list。**
   这是 Q1 坑 2 的直接后果，不是风格问题，写成字符串就崩。
4. **trial 记录里要存 `top_k`**，以及投递方式（user / tool / text）作为一等字段。

## 这份 pilot 没有回答的

- Q3 用的是「大目标 + 已核实不存在的干扰项」，是**天花板测试**：它证明模型读到了图里
  最显眼的东西，不证明 tool_result 里的图在**细粒度/意识形态相关**的信息上被读得一样充分。
- 只测了 1 张图。真正的 scheme 是 3 张（`list_dir` 出三个文件名）。多图时
  `image_count` 在模板里是全局累加的，位置编码与注意力预算都会变，需要单独验。
- 没有测 tool_call 是模型**自己发**的情形。这里所有 tool_call 都是我们预填进对话的，
  模型只是在读一段已完成的转录。
