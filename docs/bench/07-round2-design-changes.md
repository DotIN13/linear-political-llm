# Benchmark 设施 ⑦ —— 第二轮的六处设计修改

前六份文档是 Decks 板子的转换，**它们仍是设计的权威版本**。这一份不是它们的替代，
而是一张**差异表**：第一轮把 01–06 落成代码时暴露出六个问题，第二轮逐条改掉了。
凡是这里和 01–06 冲突的地方，**以这里为准**，并且每条都写清为什么。

---

## A. 候选词 → A/B 选项（推翻 04 的「两个单词的 logprob」）

**问题。**04 里 T1 写的是 `"Biden" | "Trump"` 这样两个单词直接取 logprob。
实测 tokenizer：八对候选词里**五对不对称** —— `"Biden"` 切成 `["B","iden"]`，`"Trump"` 是一个 token。
于是 `logP(Biden) − logP(Trump)` 实际比较的是「一个词首子 token」和「一个完整词」，两者不在同一个尺度上。

**改法。**问题末尾附标号选项，模型只回一个字母：

```
In the 2020 U.S. presidential election, who would you have voted for?
A. Biden
B. Trump
Answer with a single letter.
```

surface 声明 `options: list[str]`（两个语义选项），**实际测的候选 token 固定是 `"A"` / `"B"`**，
两者都是单 token，天然对称。

**顺序平衡是配套的、不是可选的。**模型对「第一个选项」有与内容无关的偏好，
所以每个条目跑两遍（`order="ab"` / `order="ba"`），`order="ba"` 的读数取相反数后与 `ab` 平均。
两者之**差**存成 `position_bias`：它是免费的健康指标，某道题差得离谱说明答案主要由位置决定。

**开跑前的候选闸门。**`bench check` 默认执行，和能力闸门放在一起：
① `"A"`/`"B"` 各自是单 token；② 用一个样例 prompt 前向一次，确认答案位置的 argmax 落在 `"A"`/`"B"` 之一；
③ 每个 surface 打印结果。argmax 落不到就说明提示词没让模型进入「回一个字母」的模式，
这个 surface 在这个后端上不可用 —— 这一条比事后看数便宜得多。

## B. `variant` 进签名（补 01 的记录格式）

```
trial_key = sha256(surface, item_id, condition, variant, adaptor, model, seed, measurement_rev)
```

`variant` 是 dict，规范化成 canonical JSON（key 排序、无空格）后进哈希，空 dict 的确定形式是 `{}`。
本轮装两样：`{"phrasing": 0, "order": "ab"}`。做成 dict 而不是两个字段，
是为了以后加「图片位置」「图片张数」这类同条目内的重复维度时不用再动签名。

**variant 空间由 surface 自己声明**（`Surface.variants()`），`bench run` 遍历它而不是靠调用方传参。
这样「三种问法都跑了」和「有人把 variant 拼错了」才分得开；`bench check` 顺带校验规范形式。
**问法的字符串本体放在 surface 里**，于是 `measurement_rev`（见 D）自动覆盖它 ——
改一个问法的措辞会让对应记录失效，这是对的。

**variant 是重复测量，不是独立观测。**3 种问法 × 2 个顺序 = 每个条目 6 行；
把行数当 n 会让 n 虚高 6 倍、p 值全错。所以 `bench score` 先在
`(surface, item_id, condition)` 内对 variant 求平均得到每条目一个值，**然后**才算均值/sd/n，
`n` 数的是条目数；表里同时打印 `n_items` 和 `n_rows`，让虚高一眼可见。
`--by variant` 才拆开看，那是诊断视图不是主分析。

## C. 基线条件单独处理（补 03 的条件 E）

条件 E 不给图、对话逐字相同，所以 300 个条目会跑出 300 份**字节相同**的对话
（第一轮实测：80 条 trial 只有 42 个唯一对话）。

`Surface.is_item_invariant(condition)` 对 choice surface 的 `"E"` 返回 True。
`bench run` 遇到 item-invariant 的条件时**每个 (surface, variant) 只跑一次**，
写一条 `item_id = "__baseline__"` 的记录；`bench score` 把它当常数基线广播给该 surface 的所有条目，
并默认多报一列 `outcome_minus_baseline`。

## D. `measurement_rev` 取代 `code_rev` 进 key（推翻 01 的 `code_rev`）

`code_rev` 用 git HEAD，于是**改 README 就作废全部数据**；而脏树的 `-dirty` 后缀又不区分具体改了什么。

`measurement_rev` = 对**只影响测量的东西**做哈希：`bench/adaptors/**`、`bench/surfaces/**`、
`bench/types.py`、`bench/store.py` 的文件内容，加上探针权重文件的 sha256，排序拼接后取前 12 位。
`code_rev`（git HEAD ＋ dirty 标记）**仍然记进记录和 manifest**，只是不参与去重 ——
它回答「这批数是哪版代码出的」，去重则由「测量本身变没变」决定。

## E. 分层序号就是自变量（推翻 03 的「三张图分数的均值」）

03 说「条目的分数取三张图的均值，测量噪声按 1/√3 缩小」。第一轮实测**不成立**：
三张图同时在场时读出的 `s_img` 不等于三张单独分数的均值，而且不对称
（stratum 0：−0.331 → −0.453；stratum 9：+0.576 → +0.325，两端都向中间塌）。
三张图放在一起会互相影响，所以「均值」不是那个被操纵的量。

**处置：分层序号 `stratum`（0–9）就是主自变量**，做序数处理，不需要精确的连续 x ——
反正三张图**本来就是从同一层抽的**，层才是真正被操纵的东西。
item 里的 `covariates.image_mean_mean` 降级为普通协变量，item schema 加显式字段 `primary_iv = "stratum"`。
实测的 `s_img` 继续记，用途变成两个：① 操作检查（`stratum` 是否单调预测 `s_img`，
`bench score` 默认打印这行）；② 备用的连续分析。

## F. 只做完整性筛选，其余全部变标注（推翻 03 的七道筛子）

03 的 F1–F6 里，`tokens:300-800` 这一道砍掉了 99385 张里的 58182 张（58.5%），
而真实分布是 min 64 / 中位 260 / max 400 —— **一张都没超过 400**，阈值本身就是错的。

但处置比「把区间改对」更彻底。**新原则：不做内容筛选，只做数据完整性筛选，其余一律变成标注列。**
理由有两条：按内容删数据（尤其「户外」「有人」「有字」这类）等于对可能在因果路径上的变量做条件，
会把机制本身控制掉；而且每道筛子的阈值都是研究者自由度，删得越多越难辩护。

- `DEFAULT_FILTERS` 只剩完整性：`image_token_mismatch == 0` 且 `num_image_tokens > 0`。
- `no_person` / `no_text_cats` / `objects:` / `aspect:` / `tokens:` 的**实现全部保留**，
  仍可通过 `--filters` 手动开启 —— 它们从主分析降级为敏感性分析。
- 原来被拿去当筛子的量全部变成 item 的标注列：`n_persons`、`has_text_cat`、
  `n_objects`、`aspect`、`num_image_tokens`。
- `items/decile_profile_v2.csv` 照旧报每层的高频类别，另加每层的 `share_with_person` /
  `share_with_text_cat`。**不做「室内/户外」这类手写映射** —— 那是另一种形式的主观介入，
  只报类别频次，让读者自己判断。

**一条必须跟着 `n_persons` 走的警告：LVIS 是 federated 标注。**
每个类别只在属于它的那个子集里被穷尽标注，`person` 只在 100170 张里的 **1928 张（1.9%）**被标了。
所以 `n_persons == 0` 的意思是「没有 person 标注」，**不是**「画面里没有人」。
这同时说明 03 的 F2（排除含 person 的图）从来就没做到它声称的事 ——
它只删掉了 2% 的图，却给人一种「这批图里没有人」的错觉。要真正控制人脸通道得跑检测器，不能靠 LVIS 标注。

---

## 这一轮**没有**改的

- **对话模板一字未动**（03 的红线：模板里不得出现政治词）。改的只是最后那一轮用户问句的形状。
- **三图仍是下限**（03 的剂量论证不受 E 影响 —— 受影响的只是「怎么给这个剂量赋一个数」）。
- **explore / confirm 切分照旧**，确认集仍然不看。v2 的采样写成新文件名（`items/explore_v2.jsonl`），
  v1 原样保留：03 说过刺激物不就地覆盖。
- **`judge` 和 `report` 仍是 stub**，`sample` 仍不做 F4（OCR 政治词表，缺依赖）。
