# Benchmark 设施 ② — CLI、目录与迁移

> 由 Decks 板子 `benchmark-设施-②-cli-目录与迁移.html` 自动转换。原板子是设计的权威版本。

# Benchmark 设施 ② —— CLI、目录，以及从现有代码怎么迁

抽象和能力模型在 **《Benchmark 设施 ① — 五个抽象与能力模型》**。

### 先说一句：这不是重写，是**把已经写好的东西拆开**

仓库里这五个抽象的实现**全都已经存在**，只是被焊死在几个大脚本里：
 `token_scoring.py` 里是 Adaptor 的激活捕获，`lvis_persona_study.py` 里是六个 Surface，
 `classifiers/*.py` 里是 Extractor，`senate_speech.py` 里是 Judge，
 `ab` 客户端就是 opencode 的 Adaptor。**下面那张迁移表是逐条对应关系**。
 真正要新写的只有 Store 的去重、CLI 的骨架，和能力闸门 —— 加起来几百行。

## CLI —— 七个子命令，每个只做一件事

### [cli]
```
# 看有什么，以及什么能配什么
bench surfaces                          # 列出 surface 及其 requires
bench adaptors                          # 列出 adaptor 及其 capabilities
bench check --surface speech --adaptor opencode
    ✗ speech 需要 {generate}            OK
    ✗ probe_points 非空但 opencode 无 activations → 该 surface 会降级运行（不产出 s_txt）

# 造刺激物（纯 CPU，可以在 jevans 分区跑）
bench sample --pool lvis --strata image_mean --bins 10 --per-bin 400 \
             --filters no_person,no_text_cats,objects:3-15,aspect:0.6-1.7 \
             --images-per-item 3 --split explore,confirm --out items/

# 跑（唯一碰 GPU / 碰网的一步）
bench run --items items/explore.jsonl \
          --surface vote2020,guns,healthcare,border,tea_coffee,cat_dog \
          --conditions A,B,C,D,E \
          --adaptor local_hf --model Qwen3-VL-8B-Instruct \
          --probe combined_ideology_headwise_linear \
          --out runs/r1/ --resume

# 补裁判（可以换裁判重跑，不用碰模型）
bench judge --run runs/r3 --surface speech --judge speech_lean_v2 --adaptor opencode

# 出数
bench score  --run runs/r2                     # 确定性解析 + 聚合
bench report --run runs/r2 --by decile --plot  # 剂量反应曲线、安慰剂、中介
```

### CLI 上三个不显眼但要紧的决定

- **`bench check` 是独立子命令。**在排队几小时之前就能确认这个组合跑得通，而不是跑到一半发现 adaptor 给不出 `s_txt`。**它也是能力闸门唯一的用户界面。**
- **`--resume` 是默认行为，不是选项。**Store 按 `trial_key` 去重，所以重复执行同一条命令永远安全。这一条决定了你敢不敢在 4 小时的 srun 里跑 48000 次前向。
- **`judge` 和 `run` 分开。**裁判改一个字就是新的 `judge_id`，缓存自然失效，但模型响应不用重跑。**裁判会改很多次**，这一刀省的是真金白银。
- **`score` 和 `report` 也分开。**score 是幂等的数据变换，report 是会反复调的展示。混在一起你会为了改一根坐标轴重跑解析。

## 目录结构

### [dirs]
```
bench/
  __init__.py
  types.py            # Item · Trial · Response · Outcome · Capability
  store.py            # append-only JSONL + trial_key 去重 + 对话全文按哈希存
  cli.py              # 七个子命令
  sample.py           # Sampler：分层、过滤、配平、explore/confirm 切分
  adaptors/
    base.py           # Protocol + capability 声明
    local_hf.py       # ← token_scoring.py 的前向与钩子
    openai.py
    opencode.py       # ← ab 客户端
  surfaces/
    base.py
    choice.py         # 二选一 logprob：vote2020 · guns · healthcare · border
    control.py        # 非政治对照：tea_coffee · cat_dog · beach_mountain
    scale.py          # 七点量表（数字 token 的 logprob）
    ranking.py        # ← lvis_persona_study.py 的六个域
    senate.py         # ← apol_to_pol_study.py 的候选人排序
    speech.py         # ← apol_to_pol_study.py 的演讲
  judges/
    base.py
    speech_lean.py    # ← classifiers/senate_speech.py 的裁判提示词
    rewrite_bias.py   # ← label_rewrites.py 的五档
  probes/             # 复用仓库现有的 probes/ 包，不重写

items/     runs/     conversations/     judge_cache/
```

### 几条约定

- **`bench/` 放在仓库里，不单独开 repo。**它依赖 `probes/`、依赖 `results/probes/` 的权重、依赖 `data/` 的分数表 —— 拆出去只会让路径变复杂。等它稳定了、别的项目要用了再说。
- **surface 和 judge 用**注册表**而不是插件系统。**一个 `@register("vote2020")` 装饰器就够了。不要写动态加载。
- **`conversations/` 按 sha 存全文。**一条 trial 记录只放哈希；对话本身（含图像路径、写死的 assistant 回合）单独存一份。这样 JSONL 保持可 grep，而对话仍然可完整复原。
- **`runs/` 里一次运行一个目录**，里面是 `trials.jsonl` ＋ `manifest.json`（命令行、code_rev、开始结束时间、adaptor 配置）。**manifest 是「这批数怎么来的」的答案**，跟 `MANIFEST.md` 那件事是同一个道理。
- **`items/` 一旦生成就不再改。**要换筛选条件就出一份新的、换个名字。**永远不要就地覆盖刺激物** —— 否则旧的 run 就失去了意义。

## 迁移表 —— 现有代码逐条对应

### [migration]
- scripts/probes/token_scoring.py→`adaptors/local_hf.py` —— 模型加载、图像编码、模块钩子、探针打分全在里面，而且已经支持 qwen3-vl / gemma4 两个家族和三种探针。**要加的只有「在指定位置读」和「取下一 token logits」。**
- probes/{base,headwise_linear_probe,…}.py→`bench/probes/` 直接 import，不动。`build_steering()` 以后给 steer 能力用。
- lvis_persona_study.py 的 DOMAIN_CONFIGS RANKING_SHORTLISTS · _make_ranking_config→`surfaces/ranking.py` —— cars / beer / weekend / books / politics / voting 六个 surface。**清单一字不改**，这样新旧结果逐项可比。
- apol_to_pol_study.py 的 phase_candidates→`surfaces/senate.py` —— 四场 2026 参议员选举、候选人简介照搬，但输出从「解析叙述」改成两个名字的 logprob。
- apol_to_pol_study.py 的 phase_speech→`surfaces/speech.py`（生成）＋ `judges/speech_lean.py`（五维裁判）。**裁判提示词一字不改。**
- classifiers/{ranking,travel,cars,beer,books,weekend}.py→各 surface 的 `extract()`。**但大部分会被 logprob 取代** —— 保留它们主要是为了跑黑盒 adaptor 时还有解析路径可用。
- classifiers/senate_speech.py→`judges/speech_lean.py` 的解析部分。
- label_rewrites.py（GPT-5 五档打分）→`judges/rewrite_bias.py`。阶段①那条「改写去偏」的线也就顺手接进来了。
- ab 客户端（agent-bridge）→`adaptors/opencode.py` ＋ `adaptors/agentbridge.py`。session 能力天然就有，agentic surface 靠它。
- extract_lvis_category_scores.py 的 bbox→token 网格→`sample.py` 的掩码映射（第二步遮挡实验要用）。
- （无对应）＋**要新写的：**`store.py` 的去重、`cli.py` 的骨架、能力闸门、`report.py` 的统计（仓库里 grep 不到 `scipy`）。

### 一周能立起来，别做成半年工程

- 第 1 天`types.py` ＋ `store.py` ＋ `local_hf` adaptor 的最小版（只要 generate ＋ logprob）。
- 第 2 天**第一个 surface 跑通**（`vote2020`），端到端出一条记录。**这一天必须见到数**，否则就是在建框架而不是做实验。
- 第 3 天探针接进 adaptor（`s_txt`），Sampler 出 items，能力闸门。
- 第 4 天其余 choice / control / scale surface —— 这些都是同一个模板换问题，很快。跑 R1。
- 第 5 天speech surface ＋ judge ＋ opencode adaptor。
- 之后`report.py` 和统计，边分析边补。

### 明确**不**要做的东西 —— 这类设施最常见的死法是变成目的本身

**不要数据库**（JSONL ＋ 文件够用到十万条以上）·
 **不要插件系统**（一个注册装饰器就行）·
 **不要 web UI**（`report` 出 PNG/PDF 就够）·
 **不要分布式调度**（一张 H200 顺序跑 48000 次前向是几小时，不是几天）·
 **不要为「以后可能有的后端」预留抽象**（现在只有四个 adaptor，等第五个出现再重构）·
 **不要在 surface 里做统计**（surface 只产出原始量，聚合是 report 的事）。
 **判据很简单：第 2 天见不到第一个数，就说明方向错了，砍掉一半重来。**

