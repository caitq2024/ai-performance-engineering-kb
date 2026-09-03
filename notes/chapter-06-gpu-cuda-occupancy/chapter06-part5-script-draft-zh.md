# 第六章 Part V 讲稿（预览稿，对应 chapter06-part5-draft.pdf，共 8 页）

> **说明**：Part V = 原书最后两个大节：Debugging Functional Correctness with NVIDIA Compute Sanitizer（p259）+ Roofline Model: Compute-Bound or Memory-Bound Workloads（p261），加全章收尾（Key Takeaways p265 / Conclusion p266）。页码为预览稿编号。
> 每节结构：🎤 口播稿（覆盖页面全部要点）／📖 页面对照（幻灯片文字的中文版）／大白话／❓ 预备问答。

---

## 第 1 页｜标题页

🎤 最后一部分。转场词（对应第 2 页顶部灰字）："接下来不能继续盲调 block size 了——我们需要一个方法判断到底撞上了 compute ceiling 还是 memory ceiling，这就是 roofline；中间先绕一小段正确性检查。"顺序和书一致：先保证算得**对**（sanitizer），再判断往哪个方向调**快**（roofline），最后收全章。

---

## 第 2 页｜Debugging Correctness: NVIDIA Compute Sanitizer（正确性调试）

🎤 为什么讲性能的章节要插一页正确性？因为 Part II 第 5 页说过：**越界是未定义行为，可能一声不吭**——十万个线程的场面里，"我这儿跑过没问题"毫无意义。Compute Sanitizer 是官方的四合一检查工具，左右两框按问题类型分组：
**内存类（左框）**：**memcheck**——抓越界/未对齐访问（全局、本地、共享都管）、硬件异常、设备内存泄漏，配 `--check-device-heap` 连设备侧堆也查——Part II 那个 `input[63]` 就归它抓；**initcheck**——抓"读了未初始化的全局内存"，书里点名经典成因：**忘了做 H2D 拷贝**（分配了、没拷、直接用）。
**并发类（右框）**：**racecheck**——抓共享内存上的数据竞争（WAW/WAR/RAW），症状是结果**时对时错**；**synccheck**——抓非法同步原语、不配对的栅栏（比如 `__syncthreads()` 写在分支里，一半线程到得了、一半到不了），症状是死锁或状态错乱。
用法一行：`compute-sanitizer [--tool 工具名] <程序>`；用 `--kernel-name` 只查目标 kernel，用 NVTX 标注区间。书的工程建议：配合 `--error-exitcode` 放进 **CI**——让正确性回归在合并前就被拦下。

📖 页面对照：
- memcheck: "out-of-bounds / misaligned accesses (global, local, shared), HW exceptions, device leaks" → 越界/未对齐/硬件异常/泄漏。
- initcheck: "reads of uninitialized global memory (classic cause: a missing H2D copy)" → 读未初始化全局内存（经典成因：漏了 H2D 拷贝）。
- racecheck: "shared-memory hazards: WAW / WAR / RAW → nondeterministic behavior" → 共享内存竞争，结果不确定。
- synccheck: "invalid sync primitives, mismatched barriers → deadlocks" → 非法同步、不配对栅栏。

大白话：**十万个线程面前，"我这儿跑过没问题"毫无意义——sanitizer 是你的安全带。**

❓ 可能被问："四个工具每次都要跑吗？"→ 日常 memcheck 最常用（越界是头号问题）；racecheck/synccheck 在用了共享内存 + 栅栏的 kernel 上跑；CI 里可以夜间全量跑。注意 sanitizer 模式下 kernel 会显著变慢，只在排查和 CI 用，别在生产开。

---

## 第 3 页｜Roofline Tells Us Which Resource Is the Limit（屋顶线模型）⭐建议讲 2–3 分钟

🎤 全章的"指南针"到了。三步讲清：
**第一步，定义横轴**：**算术强度（arithmetic intensity）= 每从 HBM 搬 1 字节，做多少次浮点运算（FLOPs/Byte）**——一个 kernel 的"体质"，和硬件无关。
**第二步，画两道屋顶**：**平的算力屋顶**（FP32 约 80 TFLOP/s——你算得再密也超不过算力上限）和**斜的内存屋顶**（约 8 TB/s——搬得再多也超不过带宽上限）。两道屋顶交汇处叫**屋脊点（ridge point）**：80T ÷ 8T ≈ **10 FLOPs/Byte**。落点在屋脊左边 = memory-bound（内存受限），右边 = compute-bound（算力受限）。
**第三步，现场算我们的案例**：C = A + B——读 A、B 共 8 字节，加 1 次，写 C 4 字节 → **1 FLOP ÷ 12 Byte ≈ 0.083**——离屋脊点差 100 多倍，**无可救药地 memory-bound**。这就从数学上解释了 Part IV 第 8 页：为什么占用率救不了它——图 6-21 上它就钉在最左边的斜坡上。
书中注：LLM **两头都占**——**prefill 偏 compute-bound**（长序列一次算完，权重充分复用），**decode 偏 memory-bound**（一次一个 token，权重复用率 1）——第 15–18 章的主线。

📖 页面对照：
- "Arithmetic intensity = FLOPs per byte moved to/from HBM" → 算术强度：每字节搬运对应的浮点运算数。
- "flat compute roof (~80 TFLOP/s FP32) and sloped memory roof (~8 TB/s), crossing at the ridge point ≈ 10 FLOPs/byte" → 平屋顶与斜屋顶交于屋脊点约 10。
- "1 FLOP / 12 B = 0.083 FLOPs/byte --- hopelessly memory-bound" → 0.083，离屋脊 100 多倍，彻底内存受限。
- "prefill tends compute-bound, decode tends memory-bound" → 预填充偏算力受限，解码偏内存受限。

大白话：**屋顶线先问一句：这个 kernel 是饿数学，还是饿数据？答错方向，白干。**

❓ 可能被问："80 TFLOP/s 哪来的？Tensor Core 不是几 P 吗？"→ 80T 是 FP32 CUDA core 的示意峰值。换用 Tensor Core + 低精度，**算力屋顶整体抬高、屋脊点右移**——所以"换精度"等于"换屋顶"，正好是下一页。

---

## 第 4 页｜Moving Right on the Roofline: Lower Precision Pays Twice（低精度一鱼两吃）

🎤 知道自己 memory-bound 之后怎么办？往屋脊点的**右边**挪——让每个字节干更多活。三条路：数据留在片上复用（Part III 的本事）、算子融合（少走几趟 HBM）、或者最直接的——**把字节变小**。
算一笔账：一次 128 字节的内存事务，能装 **32 个 FP32 = 64 个 FP16 = 128 个 FP8 = 256 个 FP4**。FP32 → FP16，同样的带宽下数据量翻倍，**FLOPs/Byte 立刻翻倍**；FP8 相对 FP16 再翻一倍。这就是"一鱼两吃"：**低精度既是算力优化（Tensor Core 低精度吞吐更高），也是带宽优化（同样字节装更多数）**——而对 memory-bound 的负载，第二吃才是主菜。
Blackwell 的硬件加持：原生 **FP8 和 FP4 Tensor Core**，外加**硬件解压**——权重以压缩形态存在 HBM 里（甚至比 FP4 更狠的 4 比特/2 比特方案），读取时硬件即时展开、再转成 FP16/FP32 做精确累加。净效果：**等效内存带宽变大**——对 memory-bound 的 token 生成是体系结构级的礼物。
蓝框要点一句话：**字节减半 → 算术强度翻倍 → 向算力屋顶靠近。**

📖 页面对照：
- "do more work per byte: reuse data on-chip, fuse ops --- or simply shrink the bytes" → 每字节干更多活：片上复用、融合、或缩小字节。
- "one 128-byte transaction carries 32 FP32 = 64 FP16 = 128 FP8 = 256 FP4" → 一次事务的装载量对比。
- "hardware decompression: weights stored compressed in HBM are expanded on the fly and cast to FP16/FP32 for accurate accumulation" → 硬件解压：压缩存储、即时展开、高精度累加。
- 蓝框 "halve the bytes → double the arithmetic intensity → move toward the compute roof" → 减半字节、翻倍强度、靠近算力屋顶。

大白话：**装货的卡车没变，把每件货做小一半，一趟就能拉两倍的活儿。**

❓ 可能被问："压缩权重会损精度吗？"→ 分两层：**量化本身**（FP16→FP8/FP4）有损，靠训练侧/校准控制；**硬件解压那一步**是对已量化数据的无损展开，且累加用 FP16/FP32 做——不额外叠加误差。

---

## 第 5 页｜Profiling Workflow: Diagnose, Fix, Re-measure（诊断—修复—复测）

🎤 全章方法论收口：怎么用 profiler 把前面所有知识串成一个循环。
**诊断特征**：memory-bound 在 ncu 里的指纹是 **DRAM 利用率高 + ALU 利用率低**——warp 都在等内存；同时留意 global load efficiency（全局加载效率）下滑——那是访存不合并的信号。nsys 时间线上 kernel 之间的**空档** = GPU 在等数据传输。
**两个惯犯**：期望"拷贝与计算重叠"却看到串行排队？① 不小心用了**默认流**（自带隐式全局同步——Part II 第 8 页为什么建非阻塞流的回收）；② **忘了用锁页内存**——没有 pinned，`cudaMemcpyAsync` 名字里有 Async 也**无法**真正重叠（Part II 第 3 页的伏笔在这兑现）。
**修复后的样子**：右图（图 6-22）上下对比——同步版"拷贝、算、拷贝"排队走；异步版三者叠着走，空档消失，访存管道利用率逼近峰值。
右下斜体是全章最后一条纪律：**每改一处就复测一次**——profiler 确认停顿真的减少了再动下一处。
工具的新特性顺嘴提：ncu 的 range replay、源码关联——知道有就行。

📖 页面对照：
- "memory-bound signature in ncu: high DRAM utilization + low ALU utilization" → memory-bound 指纹：DRAM 高、ALU 低。
- "two usual suspects: an unwanted default-stream sync, or a missing pinned-memory buffer" → 两个惯犯：默认流隐式同步、缺锁页内存。
- "without pinned memory, cudaMemcpyAsync cannot overlap with kernels" → 没有锁页内存，异步拷贝无法与计算重叠。
- "Always measure after each change" → 每改必测。

大白话：**修车不能凭感觉换零件——每换一个，上路测一圈。**

❓ 可能被问："global load efficiency 低说明什么？"→ 请求的字节数 ÷ 实际搬运的字节数低——访存不合并，warp 的 32 个请求散落在很多缓存行里，每行只用了几个字节。解法是第 7 章的 coalescing。

---

## 第 6 页｜Key Takeaways（书中八条要点）

🎤 书的八条要点，我们按四对念，每对一句话（这页 90 秒，别逐条展开——都讲过了）：
**执行模型一对**：SIMT——32 线程的 warp 齐步走，大量 warp 在飞 = 延迟被藏（Part I）；层级——线程 → block（≤1,024）→ grid，`__syncthreads()` 能少用就少用（Part I/II）。
**资源一对**：占用率与上限——取 32 的倍数；每 SM：64 warp、32 block、228 KB 共享内存、每线程 255 寄存器（Part I 上限表）；启动参数——256 起步、网格向上取整、profiling 定终值（Part II/IV）。
**内存一对**：异步分配——MallocAsync + 流 + 内存池，PyTorch 缓存分配器同理（Part II）；内存阶梯——上层复用（寄存器/共享/L1），下层合并（L2/HBM）；统一内存用 prefetch + advise 驯服（Part III）。
**方向一对**：roofline——FLOPs/Byte 决定主战场；FP16/FP8/FP4 + 硬件解压向右挪；TMEM + UMMA 能把 kernel 从内存受限拉回算力受限（Part V）。

📖 页面对照：左右两栏八条要点即上述四对的原文，逐条对应 Part I–V 的章节。

大白话：**八条要点 = 四对钥匙：怎么执行、给多少资源、数据放哪、往哪个方向调。**

---

## 第 7 页｜Conclusion and What's Next（总结与下一步）

🎤 蓝框一句话总结全章：**让 GPU 忙起来（occupancy）、让数据靠得近（memory hierarchy）、让 roofline 告诉你下一仗打哪。**
书的收尾提醒（也是第 16 页 occupancy 定义页埋的伏笔，现在收回来）：**最大占用率并不总是最优**——单线程 ILP 足够时，中低占用率也能赢；有时"线程更少、每人寄存器更多"反而更快。**一切以实测为准**——这句话是全章的底色。
预告后面的书：第 7–8 章讲 warp 分化与访存调优（我们埋的好几个"第 7/8 章见"在那兑现）、第 9 章算术强度、第 10 章 TMA 与 warp 专业化。
最后是我的预告：**我的第二讲——第 12 章**——建立在本章之上：单个 kernel 快了之后，怎么**编排**它们，让 GPU 永远不用等 CPU？到时见。
（Questions? 停在这页答疑——结论页留在屏幕上，别切到谢谢页。）

📖 页面对照：
- 蓝框 "Make the GPU busy (occupancy), make data close (memory hierarchy), and let the roofline tell you which battle to fight next." → 忙起来、靠得近、罗盘指路。
- "max occupancy is not always optimal --- with enough ILP, moderate/low occupancy can win... Always benchmark." → 满占用未必最优；ILP 足够时中低占用可赢；永远实测。

大白话：**全章三个动词：填满、靠近、看罗盘。**

❓ 可能被问："ILP 足够时低占用为什么能赢？"→ 回忆 Part I"发射 vs 执行"的问答：同一 warp 的独立指令可以背靠背发射。每线程干的独立活儿多（ILP 高），少量 warp 也能把管线喂饱，还省下寄存器让每线程拿更多——Volkov 经典结论，书里点到为止。

---

## 第 8 页｜References & Further Reading（参考文献）

🎤 不逐条念。口头一句："引用主要四类——原书第 6 章、NVIDIA 的架构调优指南和 CUDA 编程指南、roofline 原始论文（Williams 等，CACM 2009）、以及各工具的官方文档；本 deck 示意表格的真实基准数据在书的 GitHub 仓库。感谢大家，回到上一页提问。"

📖 页面对照：参考文献列表（书、NVIDIA 文档、roofline 论文、工具文档、代码仓库）。
