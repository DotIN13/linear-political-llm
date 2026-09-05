# Benchmark 设施 ⑩ —— 分层改成三个桶 ＋ 纯文本对照 ＋ S1 可视化数据

> 第四轮。把十分位分层换成三个固定阈值的桶（`<-0.5` / `[-0.5,+0.5]` / `>+0.5`），
> 采样器默认切到桶；跑了一个小 run（S1 只跑 `scheme=chat`，四个臂 A/B/C 共 22 条生成），
> 并把 A 臂 27 张图 + 两个 json 拷到 `agent-bridge-tmp/uploads/s1viz/` 供板子使用。
> 设计权威来源是两块 board（`board-buckets.html`、`board-tasks.html`）。

---

## 0. 结论（先直说，不修饰）

**A vs B 的对照给出的答案是：视觉通道有增量，且增量很大。**

纯文本臂（B，只写 LVIS 类别名）复现了方向、但没有复现幅度。A 臂（有图）的三桶
`s_gen` 跨度 **0.943**（−0.679 → +0.264），B 臂跨度只有 **0.391**（−0.289 → +0.102）。
物体身份解释了大约四成，剩下六成是视觉特有的推断——正是 board 要的那个「强主张」。

同时，第三轮的两个发现**全部复现并放大了**：`s_pre` 仍然是平的（照片的痕迹不落在前缀
末尾），裁判的 `lean` 字段 22 条里 **0 条 `lean_right`**（模型文本永远中间偏左、从不
偏右，而探针 `s_gen` 在高桶读到了 +0.264 的右移）。

---

## 1. 三个桶的 `s_gen`（A 臂）与 A vs B 的差

| 桶 | A `s_gen`（有图） | B `s_gen`（纯文本） | A − B | n |
|---|---|---|---|---|
| low  | **−0.679** | −0.289 | −0.390 | 3 |
| mid  | **−0.096** | −0.261 | +0.165 | 3 |
| high | **+0.264** | +0.102 | +0.163 | 3 |

A 臂逐条：low −0.673 / −0.695 / −0.669；mid −0.003 / −0.098 / −0.188；high +0.258 / +0.267 / +0.269。
B 臂逐条：low −0.288 / −0.436 / −0.143；mid −0.271 / −0.245 / −0.269；high +0.399 / −0.082 / −0.013。

**读数**：A 臂是一条陡峭的单调剂量曲线（跨度 0.943），B 臂方向一致但平得多（跨度 0.391），
且 B 在 low 与 mid 几乎重合（−0.289 vs −0.261），只有 high 明显转右。A−B 在两端同号：
low 端 A 更左 0.39，high 端 A 更右 0.16——视觉通道在**两个方向**上都加了增量。

---

## 2. 三个桶的 `s_pre`（A 臂）—— 还是平的

| 桶 | `s_pre` | n |
|---|---|---|
| low  | −0.349 | 3 |
| mid  | −0.331 | 3 |
| high | −0.350 | 3 |

整条量程只有 0.02（−0.349 / −0.331 / −0.350），没有任何随桶单调的趋势。第三轮在
十分位上它是平的，这轮桶更极端（low 到 −0.82、high 到 +1.09），它**仍然平**。照片的
痕迹不是写进「问题之前的那个前缀状态」里的；它是在生成过程中现取现用的。

---

## 3. C 臂 4 条基线的 `s_gen`

- 均值 **−0.215**，4 个值：**−0.175 / −0.224 / −0.223 / −0.240**（sd ≈ 0.028）。

它落在三个桶的什么位置：−0.215 在 mid（−0.096）与 low（−0.679）之间，靠近 mid 一侧。
也就是说，模型「什么都不看」的默认竞选稿本身是**中间偏左**的（−0.215）；mid 桶的图把它
往右拉了一点点（−0.096），low 桶把它猛拉到最左（−0.679），high 桶把它推到最右（+0.264）。

> 注：这轮 C 臂比第三轮的 condition E 更严格——第三轮 E 仍保留「These are some photos I
> took recently.」这句照片框架；这轮 C 是**只问 S1 问题、完全不提照片**（单轮提问）。
> 因此 C 的 `s_pre` 读的是问题末尾（+0.464），位置与 A/B 的 `s_pre` 不同，不可直接相减。

---

## 4. 投递偏移 `MAD / 图间 sd`（跨桶取图）

- **MAD / sd = 0.165**（MAD 0.069，图间 sd 0.414，n=20，跨三桶取）。
- 系统偏移：agentic 平均比 chat 偏 **−0.067**。

20 张图按 round-robin 跨 low/mid/high 取（不是全从一个桶取——第三轮第一版全取 stratum 0
导致 sd 压到 0.074、MAD/sd=1.23 的假值）。图间 sd 0.414 比第三轮跨层校准的 0.240 更宽，
因为桶比十分位两端更极端。结论不变：投递方式是一等因子，同一比较里不能把 chat 和 agentic
的探针读数直接放一起。这轮主 run 只跑 chat，所以这个偏移不影响 A/B/C 的读数本身，只是
按 board 要求每轮重测、留档。

---

## 5. 裁判 8 字段档位分布（s1_speech，gpt-5.4，strict schema）

**`lean`（主刻度）22 条里 0 条 `lean_right`。** 全部 5 个倾向维度（lean / economic /
social / foreign_policy / institutional_trust）里 `lean_right` 只出现 **1 次**（B 臂
`institutional_trust` 的 9 条里有 1 条），`far_right` 0 次。

**arm A（n=9）**

| 字段 | 档位 | 退化 |
|---|---|---|
| lean | center 6 / lean_left 3 | 是 |
| economic | center 5 / lean_left 4 | 是 |
| social | center 6 / lean_left 3 | 是 |
| foreign_policy | center 9 | 是（1 档）|
| institutional_trust | lean_left 8 / center 1 | 是 |
| formality | high 6 / neutral 3 | 是 |
| optimism | high 9 | 是（1 档）|
| concreteness | high 6 / neutral 2 / low 1 | 是 |

**arm B（n=9）**

| 字段 | 档位 | 退化 |
|---|---|---|
| lean | lean_left 6 / center 3 | 是 |
| economic | lean_left 6 / center 3 | 是 |
| social | lean_left 5 / center 4 | 是 |
| foreign_policy | center 9 | 是（1 档）|
| institutional_trust | lean_left 6 / center 2 / lean_right 1 | 是 |
| formality | neutral 7 / high 2 | 是 |
| optimism | high 9 | 是（1 档）|
| concreteness | high 7 / neutral 1 / very_high 1 | 是 |

**arm C（n=4）**

| 字段 | 档位 | 退化 |
|---|---|---|
| lean | lean_left 4 | 是（1 档）|
| economic | lean_left 4 | 是（1 档）|
| social | center 3 / lean_left 1 | 是 |
| foreign_policy | center 4 | 是（1 档）|
| institutional_trust | center 3 / lean_left 1 | 是 |
| formality | neutral 3 / high 1 | 是 |
| optimism | high 4 | 是（1 档）|
| concreteness | neutral 2 / high 2 | 是 |

第三轮的发现完整复现：**桶更极端了，裁判仍然看不到右倾**——高桶 3 条 A 臂文本被裁判一致
判成 center/lean_left，而探针 `s_gen` 读到了 +0.264 的右移。探针（内部状态）与裁判（文本）
看到的是两样东西，小模型把政治表达钳在中间偏左、从不说出来。

---

## 6. 采样器自查

**图片数 / 均值，和第 2 节那张表对得上**（99,385 张可用图，`image_token_mismatch==0`）：

| 桶 | 定义 | 实际图片数（表内） | 实际均值（表内） | 可建 item |
|---|---|---|---|---|
| low | `< −0.5` | 1,173（1,173） | −0.582（−0.582） | 391 |
| mid | `[−0.5,+0.5]` | 89,745（89,745） | +0.067（+0.067） | 29,915 |
| high | `> +0.5` | 8,467（8,467） | +0.632（+0.632） | 2,822 |

采样后（每桶 396 张、`per_bucket=400`）选中的图均值 low −0.580 / mid +0.082 / high +0.636，
与全量均值一致。高桶 top 类别 horse/airplane/propeller/sheep/zebra，低桶 umbrella/handbag/
bicycle/skateboard/backpack——与 board 的「城市步行 ↔ 牧场乡村」构念完全对上。

**`n_objects` 平衡后三个桶的均值**（board 口径的 `n_objects` = 每图 distinct LVIS 类别数，
3.9/3.7/2.7）：

| 桶 | 原始 distinct 均值 | 平衡后 distinct 均值 | n_objects（实例）中位 |
|---|---|---|---|
| low  | 3.866 | **3.899** | 8.0 |
| mid  | 3.713 | **3.760** | 7.0 |
| high | 2.733 | **2.806** | 6.0 |

平衡是把旧采样器的 `balanced_pick` 原样搬到桶上（按池内 `_objects_bucket` 份额配比），
在 high 桶里尽量多挑复杂图。它能拉高 high 桶（2.73 → 2.81）、压平差距，但**消除不掉**
——因为高桶本身 83% 的图只有 ≤4 个类别（一匹马站在草地上），这个内容差异是轴本身的性质
（board：「不可能有『探针分不同、内容相同』的图」）。残余混淆正是 B 臂（纯文本对照）要
控制的东西。详见 §8。

---

## 7. GPU 用量

- **1 个 job**（`57864680`），1 张 H200，任何时刻峰值 1 张卡。
- 总时长 **约 6.5 分钟**（05:26 → 05:33）：校准 20 张图 × 2 scheme（40 次前向）约 1 分钟，
  22 条生成约 5 分钟。
- 单条生成多数 5–6 秒，但共享节点间歇性争抢把个别 trial 拖到 ~100 秒（同 400 token，
  空闲 5s、争抢 100s，与第三轮一致）。远在 30 分钟之内，不需要拆 job。

---

## 8. 与本文件冲突 / 本文件没想到的地方

1. **`n_objects` 一词有两套意思，board 的数（3.9/3.7/2.7）是 distinct 类别数，不是
   缓存里的实例数。** `items/_cache/lvis_image_meta_v2.jsonl` 的 `n_objects` 字段是
   **实例数**（每桶 13.7 / 13.0 / 10.2，一张图可以 6 只沙丁鱼 = 1 类 6 实例），而 board
   的「平均标注类别数 3.9/3.7/2.7」是 `len(categories)`（每图 distinct 类别）。旧采样器
   的 `_objects_bucket` 分箱名「3-4 / 5-7 / 8-10 / 11-15」本来就是按类别数设计的，喂给它的
   却是实例数——这是旧代码的既有错位。这轮我**按 board 的数（distinct 类别数）平衡**，
   并同时把实例数（`n_instances`）留作标注。manifest/s1viz 里 `n_objects` = distinct 类别数，
   `n_instances` = 实例数，两个都在，字段名照任务书。

2. **`n_objects` 平衡消不掉 high 桶的「更简单」——而且本来就消不掉。** 平衡后三桶仍
   3.90 / 3.76 / 2.81（low−high 差 1.09 vs 原始 1.13），因为 high 桶里只有 14.3% 的图有
   5–7 个类别、2.1% 有 8–10 个，配比到池内份额的物理上限就这么多。这不是采样器没做好，
   是「探针轴 = 城市↔乡村」的固有内容混淆。board 自己在 reframe 卡里说过这个，而 B 臂
   （纯文本）正是它的解法——见 §1。

3. **`measurement_rev` 没有变（`4dc1055f6699`，与第三轮逐字节相同）。** 这验证了 board
   的预测：只改 `sample.py` 不影响测量哈希，测量没变、设计变了。旧 52 条 pilot 的
   `trial_key` 不会与新 run 冲突，也不会被合池（item_id 全变）。

4. **C 臂「完全不提照片」≠ 第三轮的 condition E。** 第三轮 E 是「相同对话、去掉图」，
   仍含照片框架句。任务书这轮要「完全不提照片」，所以 C 臂是**单轮只问 S1 问题**（无任何
   照片框架）。副作用：C 的 `s_pre` 读在问题末尾（+0.464），与 A/B 的 `s_pre`（前缀末尾）
   位置不同，不能直接比较；本报告只对 C 报 `s_gen`。

5. **`do_sample=False`（greedy）下 4 个 seed 仍给出 4 条不同文本。** 探针 prefill 读数
   逐字节相同（s_pre 全部 +0.4639），但 400 token 的贪婪解码在 decode 步上因为 CUDA 非
   确定性而发散（4 条 text sha 互不相同，s_gen −0.175 … −0.240）。这恰好是任务书要的
   「4 条基线有 n=4 的展布」，但也意味着 A/B 每条只跑 1 次、单条读数带 ~0.03 的噪声；
   对 0.94 的主效应无影响。

6. **B 臂 high 桶的三条极不一致（+0.399 / −0.082 / −0.013）。** hi_00028 的类别是
   boot/cow/horse/saddle/stirrup/zebra 等 10 个「牧场乡村」词，纯文本就推到了 +0.399；
   而 hi_00060（cow/giraffe/zebra，3 个词）和 hi_00062（Lego/apron/boat/sheep/toy 等，
   有玩具/船/甜甜圈这些中性词）几乎没动。这提示「类别名本身的乡村负载」就是 B 臂信号的
   来源，A−B 的差才是视觉特有部分——和 board 的对照逻辑一致，但说明 B 臂对类别清单的
   具体词高度敏感，正式跑的时候 B 臂应该逐词报，别只看桶均值。

---

## 9. 验收对照

1. 已有测试全绿 + 新测试：**204 通过**（原 197 + 新增 7，`test_sample_buckets.py`）。
2. 采样器默认切桶：`bench sample` 默认 `--strata buckets`；`--strata deciles` 保留旧路径，
   `make_items` / `write_decile_profile` 未动。
3. `item_id` 带桶名（`lvis3_lo_00000` / `_mid_` / `_hi_`），`primary_iv`=`bucket`，序数 −1/0/+1。
4. 每条记录都有 `probe.top_k`（k=16 写进 `measurement_rev` 的 note，k=8 另存 `probe.k8`）。
5. A/B 臂用同一批 9 个 item（`items/pilot_round4.jsonl`），对照成立。
6. `git diff` 只碰 `bench/` + `docs/bench/`；`data/lvis_persona/*.jsonl` 仍未被跟踪。

## 10. 复现

```bash
# 采样（CPU，登录节点）：构建桶 items + 自查
python -m bench.cli sample --strata buckets --per-bucket 400 --suffix bucket_v1 --out items/

# 取 9 个 pilot item（每桶 3 个）
python -m bench.pilots.round4_pilot --phases items

# 校准 + 22 条生成（GPU，1 卡，约 7 分钟）
sbatch bench/pilots/round4_run.sbatch

# 裁判（登录节点，走 OPENAI_API_KEY，gpt-5.4）
python -m bench.cli judge --run runs/pilot_round4 --surface s1_speech --seed 42

# 分析 + 生成 s1viz 产物（拷 27 图 + manifest.json + s1viz.json）
python -m bench.pilots.round4_pilot --phases analyze
```

原始数据在 `runs/pilot_round4/`（`trials.jsonl`、`judges.jsonl`、`calibration.json`、
`RESULTS.md`），不提交（`.gitignore`）。s1viz 产物在
`/project/jevans/tzhang3/agent-bridge-tmp/uploads/s1viz/`（27 张图 + `manifest.json` + `s1viz.json`）。
