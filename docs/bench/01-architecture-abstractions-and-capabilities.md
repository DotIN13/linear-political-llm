# Benchmark 设施 ① — 五个抽象与能力模型

> 由 Decks 板子 `benchmark-设施-①-五个抽象与能力模型.html` 自动转换。原板子是设计的权威版本。

# Benchmark 设施 ① —— 五个抽象，一个能力模型

CLI、目录结构和从现有代码的迁移映射在 **《Benchmark 设施 ② — CLI、目录与迁移》**。
 这一块只管：切成哪几个东西，以及为什么切在那儿。

### 整个设计的支点：**后端的能力不一样，所以能力必须进类型系统**

本地 Qwen3-VL 能给你 logprob 和激活；`opencode` 两样都给不了 —— 它是 API 模型上的 agent，只有文字进、文字出。
 如果 adaptor 假装它们一样，那么「这个 surface 在这个后端上跑出来的数到底是什么」就永远说不清。
 所以：**每个 adaptor 声明它能产出什么，每个 surface 声明它需要什么，CLI 在开跑前就拒绝不匹配的组合**。
 这一条决定了后面所有接口的形状，也决定了 opencode 在这套设施里该待在哪儿（**judge、造刺激物、黑盒复现臂、agentic surface** —— 不是主测量）。

### 数据怎么流

> _[图]_ 流水线：Sampler 产出 Item，Surface 把 Item 变成 Trial，中间有一道能力闸门检查 adaptor 是否满足 surface 的要求，Adaptor 执行 Trial 产出 Response，Extractor 或 Judge 把 Response 变成 Outcome，最后写进 Store。Judge 有自己的缓存，Probe 挂在 Adaptor 上。

## 五个抽象，各自的接口

### [abstractions]
- Item一份刺激物：`item_id · images[] · image_scores[] · decile · covariates`。由 Sampler 产出，**写死在磁盘上不再变** —— 它是可复现性的锚。
- Surface一个任务。**它只做一件事：把 Item ＋ 条件变成一段完整对话，并声明要测什么。**

```
class Surface(Protocol):
    name: str
    requires: frozenset[Capability]
    conditions: list[str]              # 例如 ["A","B","C","D","E"]
    def build(item, condition) -> Trial
    def probe_points(trial) -> list[ProbePoint]   # 在哪些位置读探针
    def extract(resp) -> Outcome | NeedsJudge     # 能确定性解析就别用 judge
```

- Adaptor一个后端。**它不知道任务是什么，只知道怎么把一段对话变成一个响应。**

```
class Adaptor(Protocol):
    name: str
    capabilities: frozenset[Capability]
    def run(trial: Trial) -> Response
# Response: text? logprobs? activations? session_log? usage/cost
```

- Judge自然语言 → 量化。**它自己也是一次 LLM 调用，所以它自己也走 Adaptor** —— judge 可以跑在 opencode 上，主实验跑在本地模型上。

```
class Judge(Protocol):
    judge_id: str        # = hash(prompt + model + schema)，进记录
    schema: dict         # 结构化输出
    def score(resp: Response) -> dict[str, float]
```

- Storeappend-only JSONL ＋ 内容寻址的去重键。**不要数据库。**  `trial_key = sha256(surface, item_id, condition, adaptor, model, seed, code_rev)` —— 重跑时已完成的直接跳过。  48000 次前向必然中途崩，这一条不是优化，是必需品。

### 为什么切在这几刀上

- **Surface 不碰模型。**所以「同一个任务在不同后端上跑」是免费的 —— 这正是你要的：本地 Qwen3-VL 拿探针读数，gpt-4o-mini 当黑盒复现臂，两边用**同一段对话**。阶段④和新实验之所以不可比，就是因为对话是各写各的。
- **Adaptor 不碰任务。**所以加一个新模型不需要碰任何 surface。
- **Judge 独立于 Surface。**一个 judge 可以服务多个 surface（演讲、改写、自由回答都能用同一个「政治倾向五维」裁判），而且 **judge 换了不用重跑模型** —— 响应已经在 Store 里。
- **Extractor 优先于 Judge。**能用正则或 logprob 解决的，绝不调 LLM。阶段④ travel 那个 40/40 全空，就是因为把该确定性做的事交给了脆弱的解析 —— **接口上把 `extract` 放在前面，是一个提醒**。

## 能力矩阵 —— 这张表就是闸门的实现

### [capmatrix]
| Adaptor | generate | logprob | activations | steer | images | session/tools |
|---|---|---|---|---|---|---|
| `local_hf` · Qwen3-VL · Llama-3.2-Vision | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| `openai` · gpt-4o-mini · gpt-5.4-mini | ✓ | 受限 | ✗ | ✗ | ✓ | ✗ |
| `opencode` · 经 agent-bridge · deepseek | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| `agentbridge` · claude / 任意远端 agent | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |

### 那 opencode 在这套设施里干什么

它给不了 logprob 也给不了激活，所以**不是主测量的后端**。但它有四个真实用途，每一个都值回票价：

**① Judge。**裁判就是文字进、标签出，任何后端都行。opencode 走 `ab`、跑 deepseek，**便宜且已经接通**。900 次裁判调用的成本可以忽略。

**② 造刺激物。**生成问题的改写、非政治对照题、演讲提示的变体。阶段④本来就是拿 LLM 干这个的。

**③ 黑盒复现臂。**同一个 surface 在本地模型和 API 模型上各跑一遍 —— **这正是让阶段④的旧结果和新结果可比的东西**。

**④ Agentic surface。**模型带工具、多步行动之后再问政治题（「帮我查一下附近的餐馆再推荐」）。这类 surface 只有 session 能力的后端才跑得了，**而它是最接近真实产品场景的一类**。

## 一条记录长什么样 —— 设施的成败其实在这里

### [record]
```
{
  "trial_key": "sha256:9f2c…",              // 去重键；重跑时已存在就跳过
  "run_id": "r2-explore-20260905",
  "code_rev": "5d2ce41",                    // git HEAD，进记录
  "surface": "vote2020", "condition": "C",
  "item_id": "lvis3_00417", "decile": 9,
  "images": ["train2017/000000123456.jpg", …], "image_scores": [0.51, 0.47, 0.55],
  "adaptor": "local_hf", "model": "Qwen3-VL-8B-Instruct", "seed": 42,
  "conversation_sha": "sha256:…",           // 完整对话另存一份，这里只放哈希
  "response": { "text": null,               // 这个 surface 只要 logprob
                "logprobs": { "Biden": -1.83, "Trump": -2.94 },
                "usage": { "prefill_tokens": 1421 } },
  "probe": { "probe_id": "combined_ideology_headwise_linear",
             "s_txt": 0.214, "s_img": 0.486, "s_obj": null },
  "outcome": { "kind": "logprob_diff", "value": 1.11 },
  "judge": null,                            // 这条不需要裁判
  "timing": { "ms": 183 }, "cost_usd": 0.0
}
```

### 这条记录里每一样都有理由

- **`trial_key` 和 `code_rev`。**断点续跑 ＋ 「这个数是哪版代码出的」。48000 次前向一定会崩一次以上。
- **`conversation_sha` 而不是全文。**对话里有上千个图像 token 的引用，全文进每一行会让 JSONL 爆掉。全文单独存一份，按哈希索引。
- **`probe` 和 `outcome` 分开。**探针读数是中间变量、outcome 是因变量 —— 中介分析需要它们并排躺着，而不是分散在两个文件里。
- **`judge` 是可空的、后填的。**先跑完模型，再补裁判；换裁判不用重跑模型。
- **`cost_usd` 从第一天就记。**否则半个月后没人知道这套东西花了多少。

