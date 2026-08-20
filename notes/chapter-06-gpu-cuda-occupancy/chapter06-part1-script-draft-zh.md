# 第六章 Part I 讲稿（预览稿，对应 chapter06-part1-draft.pdf，共 21 页）

> **说明**：这是 Part I 重构的预览稿。页码为预览稿编号（1–16 为你定的结构，17–21 为建议保留的 Part I 收尾内容）。确认后整合进 v2 全稿，届时统一重编页码、路线图和时间分配。

---

## 第 1 页｜标题页

🎤 大家好，今天我讲第六章：GPU Architecture, CUDA Programming, and Maximizing Occupancy——GPU 架构、CUDA 编程与最大化"占用率"。前五章讲的是系统层面（硬件、网络、存储），从这章起全书进入 GPU 内部。这一章是第 7 到 12 章所有 kernel 级优化的**地基**，所以今天概念比较多，我会全程用大白话解释，讲一个小时左右。

📖 核心术语先混脸熟：**Occupancy**（占用率）= GPU 的"工位坐满率"；**SM**（Streaming Multiprocessor，流式多处理器）= GPU 的"车间"；**Warp**（线程束）= 32 个线程绑在一起执行的最小调度单位；**Kernel**（核函数）= 跑在 GPU 上的函数。

---

## 第 2 页｜What This Chapter Covers（本章讲什么：按原书引言的编排）⭐

🎤 这一页就是原书引言给出的路线，我们全程照它走，一共四步：
① **SIMT 执行模型**——warp、thread block、grid 这套层级怎样把你的算法**映射到一个个 SM 上**。这是 Part I 的主线：Part I 的所有内容（线程层级、warp、SM 内部、分化、硬件上限）都是在展开 SIMT 这一个模型。
② **CUDA 编程模式**——kernel 怎么写、启动参数怎么定、内存怎么异步分配（Part II）。
③ **内存层级**——寄存器堆、共享内存/L1、L2、HBM3e，外加异步搬数据的硬件：TMA 和给 Tensor Core 当累加器的 TMEM（Part III）。
④ **Roofline 分析**——判断 kernel 是**算力受限**还是**内存受限**，把系统推向理论吞吐峰值（Part IV–V）。
🗣 顺便回答"章标题里的 Maximizing Occupancy 放在哪"（对应页面下方的蓝框）：occupancy 是把 ① 和 ④ 连起来的**工作指标**——足够多的常驻 warp 让 SM 在等数据时也有活干；Part IV 会用 22 倍案例把它落地。
🗣 收束条：**本章的弧线——GPU 怎么执行 → 你怎么编程 → 数据住在哪 → 什么限制了速度。**

📖 页面对照：四条编号项对应原书引言的四句话；蓝框解释章标题中 occupancy 的位置。

📚 原书精读：
> "Unlike CPUs, which optimize for low-latency single-thread performance, GPUs are throughput-optimized processors built to run thousands of threads in parallel."
> 与优化单线程低延迟的 CPU 不同，GPU 是为并行运行几千个线程而生的吞吐量优化型处理器。

---

## 第 3 页｜Roadmap: Five Parts, Each Answers One Question（路线图：五部分各答一问）

🎤 这次路线图不是目录，而是**问题清单**——表格右列就是每个部分要回答的问题：**Part I GPU 架构**（4–20 页）回答"SIMT 模型怎样把 warp/block/grid 映射到 SM 上"（中间我自己画了两张比喻图帮大家建立直觉）；**Part II CUDA 编程**（21–28）回答"如何把一个计算映射到 GPU 线程上"；**Part III 内存层级**（29–37）回答"数据放在哪里、搬动它的代价是什么"；**Part IV Occupancy 实战**（38–46）回答"更多 warp 什么时候真能提速"；**Part V 正确性与 Roofline**（47–50）回答"下一步该优化计算还是内存"。
🗣 五部分的关系一句话：Part I 讲 **SIMT 硬件怎样执行**并行工作，Part II 讲**程序怎样描述**并行工作，Part III 讲**数据怎样供应**这些计算，Part IV 讲**更多并行度何时还有帮助**，Part V 讲**下一个瓶颈在哪里**。
🗣 关键一句：后面出现 SFU、TMEM、Unified Memory 这类细节页时，请大家随时问自己"它在回答哪个问题"——答案一定在这五个里。
全章一句话（金句块）：**Expose enough parallelism, keep data close, and profile the real bottleneck.——制造足够的并行，让数据靠得近，用 profiler 找到真正的瓶颈。**

## 第 4 页｜GPUs Optimize Throughput; CPUs Optimize Latency

🎤 CPU：核少、缓存深、单线程快。GPU：一百多个 SM 并行跑几千个线程。右图（Figure 6-1）是最基本的协作流程：host（CPU 侧）把数据拷到 GPU 显存、启动 kernel、拷回结果——注意有两次跨设备拷贝。GPU 的优势来自**大规模并行**：大量 SM 和线程同时处理不同的数据，提高整体吞吐；在每个 SM 内部，多个常驻 warp 还能填补访存和流水线停顿的空档。（注意区分：SM 内的 warp 调度隐藏的是 **HBM 访问、数据依赖、流水线停顿**；CPU↔GPU 的拷贝要靠 pinned memory + `cudaMemcpyAsync` + stream 与计算重叠——那是第 27、50 页的内容。）它的甜区是**数据并行**工作：矩阵乘、卷积——同一条指令作用于海量元素；可以直接写 CUDA C++，也可以经由 PyTorch 或 OpenAI Triton 这类 Python 系工具间接生成。
大白话：**CPU 是把单个零件做得快；GPU 是一座工厂——你的任务是让每个车间都忙起来。**

📚 原书精读：
> "GPUs rely on massive parallelism to hide data-transfer latency." → GPU 依靠海量并行隐藏数据传输延迟。
> "Each GPU comprises many SMs, which are roughly analogous to CPU cores but streamlined for parallelism." → 每块 GPU 有许多 SM——粗略类比 CPU 核，但为并行精简。

## 第 5 页｜Inside a Blackwell SM: The Resource Budget（SM 的资源账本）

🎤 先打开一个 SM，看它的资源账本。这里会先碰到一个新词 **warp**：32 个线程组成的执行小组——书里也是先用后讲，先记住这个定义，第 13 页正式展开。账本上三组数字要记住：
① 每 SM 同时跟踪 **64 个 warp = 2048 个线程**——这是调度器来回切换的"池子"；
② **64K 个 32 位寄存器**（共 256 KB），单个线程最多用 **255 个**；
③ **256 KB 统一的 L1/共享内存**，其中最多 **228 KB** 可配成用户管理的共享内存（实际可用 **227 KB**——CUDA 每 block 保留 1 KB）。
正是这些大额片上预算，让一个 SM 能"玩杂耍"般同时伺候几千个线程而不用频繁下 DRAM。右图（Figure 6-2）就是 SM 的内部结构，下一页细看。
大白话：**SM 是一个车间：寄存器是每人的工具腰带，共享内存是中间的工作台。**

❓ 可能被问："227 和 228 怎么回事？"→ 共享内存可配上限 228 KB，但 CUDA 每 block 保留 1 KB，所以单 block 最多申请 227 KB。

❓ 可能被问："执行到底发生在哪？kernel 上还是 warp 上？"→ 都不是——**执行发生在 SM 分区里的物理执行单元上**（图 6-2 里的绿色格子：INT32/FP32 ALU、LD/ST、SFU、Tensor Core）。kernel 是菜谱（指令文本），warp 是厨师班组（谁），执行单元是灶台（在哪）：调度器每拍安排一个班组去灶台做菜谱的下一步。顺带纠正："CUDA core"不是 CPU 那种核，只是**一条 FP32 算术车道**，没有自己的取指译码；一个 warp 执行一条 FP32 指令占 32 条车道。车道少的单元 warp 要分几拍过——SFU 每分区只有 4 条，32 个请求分 8 拍流完，这就是 SFU 吞吐低的物理原因。

## 第 6 页｜SFUs and Load/Store Pipelines（特殊功能单元与访存管线）

🎤 SM 里还有两类容易被忽略的部件。左边：**SFU（Special Function Unit，特殊功能单元）**，专算超越函数——sin、cos、倒数、开方。关键点：它有**自己独立的管线**，不占核心"数学+访存"管线的发射名额（"双发射"机制第 8 页细讲）——慢速复杂运算永远不会堵住核心管线，混合运算的 kernel 因此有更多指令级并行。
右边：**LD/ST（load/store）访存管线**，每 SM 共 16 条（每调度器 4 条），负责读写 L1/共享内存、L2 和全局显存。书里特别警告：**具体管线数量和配对规则不受保证**——判断 kernel 是"访存发射受限"还是"计算发射受限"要靠 profiling，细节查 Blackwell tuning guide。
大白话：**SFU 是商店后面的专柜——复杂业务去那儿办，快速通道保持流动。**

## 第 7 页｜Analogy: The GPU as a Factory（比喻：GPU 工厂全景图）⭐

🎤 一间车间看完，拉远看全厂——这页就一张图、一句话：**GPU 是一座工厂**。顺着图把两套词汇立起来（口头讲，页面上不写字）：
**任务一侧**：grid = **这一整批订单**；block = **一个班组**——每个班组被安排进一间车间（SM），整个生命周期不换车间，但一间车间通常**同时容纳好几个班组**；warp = 车间把班组每 32 人编成的**执行小队**；thread = **一名工人**。
**数据一侧**：Register = 随身工具箱，L1 = 车间临时货架，Shared Memory = 公共工作台，**L2 = 全厂中转仓，HBM = 厂外大仓库**——一次全局读取依次尝试 L1 → L2 → HBM，数据原路返回。
🗣 工具箱补一句（后面 occupancy 和 spilling 都靠它）：**工具箱不是工人自带的，是从车间那块大柜子（register file，每 SM 64K 个寄存器）里静态划走的一格**——block 进驻时划走、干完才归还。所以：常驻 warp 切换零成本（工具箱一直在车间里）；每人工具箱越大、车间站的人越少（寄存器 × 线程数 ≤ 柜子容量，occupancy 权衡的物理来源）；装不下的溢出到城另一头的自助仓储（local memory，落在 DRAM）= spilling。
❓ 较真的听众可能问："register file 是四个分区共享的吗？"→ 严格说每个分区各有一块（每分区 16K、共 64K），warp 归哪个分区、寄存器就在哪个分区的柜子里；书里只讲到"64K per SM"这个粒度，回答到这里即可。
🗣 收一句："这张图后面每个部分都会回来用：Part I 讲车间和小队，Part III 讲中转仓和大仓库。"

❓ 可能被问："block 会不会跨车间？"→ 不会，从生到死在同一个 SM 上；但 block 和 SM 不是一一对应。

## 第 8 页｜Four Warp Schedulers, Dual Issue（四调度器、双发射）⭐

🎤 每个 SM 其实是**四个"迷你 SM"**：四个独立 warp scheduler（warp 调度器），各带自己的派发逻辑。两层机制：
① 每个调度器每拍发射一个 warp 的指令 → **每拍最多 4 个 warp 同时推进**；
② **dual-issue（双发射）**：同一拍里同一个 warp 可以同时发一条算术指令（INT32/FP32/Tensor Core）+ 一条访存指令（load/store）。**限制：必须来自同一个 warp，不能跨 warp 拼**。
最好情况：每拍 **4 数学 + 4 访存**指令齐飞。右边表格（Table 6-1）是每拍的上限。注意表格下的小字：书里所有指标表的数值都是**示意值**，真实 benchmark 在配套 GitHub 仓库——这是全书的统一免责声明。
大白话：**同一层楼四名调度员，每拍各派出一支小组——而且小组可以一手计算、一手取料。**

📚 原书精读：
> "You can think of the SM as four 'mini-SMs' sharing on-chip resources." → 把 SM 想成四个共享片上资源的迷你 SM。
> "Note that the dual-issue must come from the same warp—and not across warps." → 双发射必须来自同一 warp。

❓ 可能被问："发射一个 warp 和 warp 正在执行计算有什么区别？"→ **发射（issue）**是调度器的动作：一拍递一条指令进管线，递完就去管别人；**执行（execute）**是功能单元的活儿：指令在管线里流若干拍才出结果（算术约 4 拍、访存几百拍）。所以"每拍最多发 4 个 warp"≠"只有 4 个 warp 在干活"——某一瞬间几十个 warp 的指令同时在各条管线里流动。车间话术：调度员每拍只递 4 张工单，但车间里几十支小队正在工位上加工半成品。图 6-7 每一行画的是一次**发射**事件，不是执行全程。进阶：下一条指令若不依赖上一条的结果，可以背靠背发射（ILP）——这就是结尾"ILP 足够时低占用率也能赢"的原理。

## 第 9 页｜Analogy: One SM Is a Workshop（比喻：SM 车间图）⭐

🎤 前面几页的数字有点抽象，我画了一张图，把整个车间装进一页。我们把一个 SM 想成一间大车间。车间里有**四个调度分区**，形象地叫 4 个 mini-SM——注意它们**不是**四个独立的 SM，CUDA 程序只能看到整个 SM，不能指定"把这个 block 放到 mini-SM 2"；mini-SM 只是帮助理解内部调度的概念。每个分区有自己的 **Warp Scheduler**，相当于一名调度员。
货架上的**安全帽**：一顶安全帽代表一个完整 warp 的执行状态（不是一个线程）。每个分区约 16 顶 × 4 个分区 = 整个 SM 最多 **64 个常驻 warp** = 2,048 线程。"常驻"的意思是寄存器、状态都已分配好、随叫随到——**不等于 64 个 warp 同一拍都在执行**。
货架下面**被高亮的工人小组**：这一拍被调度员选中（发射指令）的 warp——4 名调度员每拍最多各挑 1 个，所以图中间写"每拍最多选中 4 个 warp"。
左下角单独放大：**1 warp = 32 个线程**，SIMT 下同一条指令、32 个不同的 idx，调度员永远整组整组地派工。
图最底部是共同的地基——Register File、L1、Shared Memory——说明四个分区仍然共用同一个 SM 的片上资源。
大白话：**64 支候选队伍待命，4 名调度员，每拍最多派出 4 支——每支队伍 32 名工人。**

❓ 可能被问："mini-SM 是真实硬件吗？"→ 是真实存在的调度分区（processing block），各有独立的 warp scheduler 和执行单元，但对 CUDA 编程模型不可见、不可指定，所以说它是"理解硬件调度的概念"。
❓ "16 顶安全帽是精确值吗？"→ 64 ÷ 4 的平均示意；实际驻留分布由硬件决定。

## 第 10 页｜The Thread Hierarchy: Threads → Blocks → Grids（线程层级）

🎤 CUDA 把并行工作组织成三层（右图 Figure 6-3）：**thread（线程）**——处理一个数据元素的工人；**thread block（线程块，又名 CTA，协作线程阵列）**——最多 1024 线程一组，组内共享快速的片上共享内存；**grid（网格）**——一次启动的全部 block，尺寸设对可以扩展到几百万线程、kernel 一行不改。调度和分发由 CUDA 运行时（以及 PyTorch）自动完成。
大白话：**工人组成班组，班组内交流便宜；公司按活儿多少雇任意多个班组。**

📚 原书精读：
> "By sizing your grid appropriately, you can scale to millions of threads without changing your kernel logic."
> grid 尺寸设对，可扩展到几百万线程而不改 kernel 逻辑。

## 第 11 页｜Blocks Cooperate Inside, Stay Independent Outside（块内协作、块间独立）

🎤 两条规则一正一反。**块内**：线程用共享内存交换数据、用 `__syncthreads()` 同步——这是个 barrier（栅栏），所有人到齐才继续。但**每个 barrier 都有开销**，书里明确说：**尽量减少同步点**（右图 Figure 6-6）。
**块间**：完全独立、执行顺序不保证任何东西。这个"不方便"恰恰是 CUDA 可扩展性的来源——调度器可以把 block 随意撒到所有 SM；你的代码在未来 SM 更多的 GPU 上**不改就能跑**。
大白话：**班组碰头会（barrier）有用但贵，能少开就少开；而且永远别假设 A 班组比 B 班组先干完。**

❓ 可能被问："barrier 让 block 等待，和 warp 切换有什么关系？"→ 在硬件眼里，"在 barrier 等队友"和"等全局内存/等数据依赖"是同一种状态：这个 warp 暂时**没就绪**。调度器的反应永远一样——跳过它、发射别的就绪 warp（同 block 的，或**同 SM 上其他 block 的**；barrier 只拦自己 block）。注意：切换救的是**机器利用率**，不是你的 block 耗时——block 还是要等最慢的 warp 过线，所以"少同步"的忠告依然成立。

## 第 12 页｜Thread Block Clusters and DSMEM（线程块簇与分布式共享内存）

🗣 **本页非本期重点，30 秒带过，标了 \*。**

🎤 传统上不同 block 的线程不能直接协作，现代 GPU 打破了这一点：**thread block cluster（线程块簇）**——一组能**跨 SM 通信**的 block，有簇级硬件 barrier。底层是 **DSMEM（分布式共享内存）**：把参与簇的各 SM 的共享内存 bank 用**片上高速互连**连成一个池子（右图 Figure 6-5）。效果：不同 block 的线程能以**片上速度**读、写、原子更新彼此的共享缓冲——**不花全局显存带宽**。这是今天大矩阵乘、LLM 负载的关键使能技术，第 10 章细讲，今天知道它存在即可。
大白话：**相邻班组在工作台之间的墙上开了个门，零件直接递过去，不用再走仓库。**

📚 原书精读：
> "This unification allows threads in different blocks to read, write, and atomically update one another's shared buffers at on-chip speeds—and without using global memory bandwidth."

## 第 13 页｜Warps and SIMT: 32 Threads in Lockstep

🎤 上一页的三层是**软件视角**；硬件看到的还有一层：block 会被再切成 **warp，固定 32 个线程一束**，在 **SIMT**（single instruction, multiple threads，单指令多线程）模型下**锁步（lockstep）执行**——32 个人同一拍做同一个动作。三个要点：① **硬件真正调度的单位是 warp，不是单个线程**——这是理解 GPU 的关键一跳；② warp 的大小**每一代 GPU 都是 32**，这个数可以焊死在脑子里；③ 硬件靠**快速切换 warp** 来隐藏长延迟事件（全局加载、缓存填充、管线停顿）。
大白话：**warp 是 32 人的施工小组——同一条指令全组一起动，否则整组等着。**
有了 warp 这个词，我们就可以打开 SM 看内部了——接下来三页全在 SM 里面。

## 第 14 页｜Why SIMT Matters: One Instruction Drives 32 Lanes（SIMT 的意义）⭐

🎤 上一页说 warp 32 人齐步走，这一页回答"**为什么要这样设计**"。看图 6-7：warp scheduler 每拍发**一条指令**，整个 warp 的 **32 条通道**同时执行它，各自处理**不同的数据元素**——图里每一行就是"一条指令、整组推进"。
🗣 **读图法（重要）**：注意左列的指令编号——Warp 8 的 instruction 11 和 instruction 12 **不是背靠背的**，中间插着 warp 2、warp 14 的指令。为什么 warp 8 歇了几拍？因为它的下一条指令还没就绪（在等内存、等依赖、或在 barrier 处等队友）。**这些"插队"就是延迟隐藏本身**：一个 warp 等待的空拍，被别的 warp 的工作填满。图里没标停顿原因，所以要靠编号读出来——第 17 页会把这个过程逐拍摆开。
好处：取指、译码、调度这些控制成本，**一条指令摊给 32 个线程**——控制逻辑做小了，省下的硅片面积全部给了算术单元。这就是 GPU 能塞下几千个"核"而 CPU 不能的根本原因。
也因此，**硬件调度的最小单位是 warp 而不是线程**：线程从不单独前进，整个小队一起走。
代价：32 人绑在一起，想走**不同的路**怎么办？——下一页。
大白话：**一张工单同时指挥 32 名工人——管理便宜；大家意见一致时威力大，意见不合时就尴尬。**

## 第 15 页｜Warp Divergence: When if/else Splits the Crew（warp 分支分歧）⭐

🎤 尴尬的情况就是 **warp divergence（warp 分支分歧）**。看左边代码：`if (i % 2 == 0) 做任务A; else 做任务B;`。同一个 warp 里：**16 个线程想走 A，16 个想走 B**。但它们原则上必须一起执行同一条指令，所以 GPU 只能：**先执行 A、把走 B 的 16 个线程屏蔽掉；再执行 B、把走 A 的 16 个屏蔽掉**——总时间 ≈ 两条路径之和，效率直接掉一半；分支路径越多，串行化越狠。图 6-8 就是这个对比：左边分叉（红蓝交错、走走停停），右边整齐一致（全速前进）。
两个要点：① **跨 warp 没有惩罚**——不同 warp 各走各的分支完全免费；② 修法：让分支条件按 **warp 对齐**（`i / 32`，一整队都走同一边），而不是按奇偶（`i % 2`，队内劈两半）。
检测靠 profiler（第 8 章 Nsight Compute）。
大白话：**同一小组一半人接了 A 工单、一半接了 B 工单，全组就得把两份活各干一遍。**

## 第 16 页｜Worked Example: One Million Adds Through Both Views（实例：一百万次加法）⭐

🎤 Part I 收尾，用一个例子全部串起来。先垫一句：**这个三尖括号语法是 Part II 的正题，这里先当"读法"看**——尖括号里第一个数是 block 数、第二个是每 block 线程数。左栏是**你写的订单**：`add<<<3907, 256>>>`——1 个 grid；**3,907 个 block**（一百万除以 256 向上取整）；每 block **256 线程**——到这里都是你选的；硬件接手把每个 block 切成 **8 个 warp**。每个线程只说一句话："我处理第 idx 个元素。"
右栏回答"**为什么是 256**"（best practice 的完整理由）：① **32 的倍数**——恰好 8 个整 warp，没有半空 warp 白占调度位；② **离 1,024 上限足够远**——每线程寄存器够用、block 的共享内存份额不大；③ **够小、能叠**——一个 SM 能同时放好几个 256 线程的 block → 常驻 warp 多 → 延迟隐藏有本钱；④ **不是魔法数**——128–512 都合理，最终靠 profiling 定。
🗣 口头补充硬件怎么消化这 3,907 个 block（页面上不写）：它们**不会同时进 GPU**——假设 4 个 SM、每个放 2 个 block：先进 8 个，每 SM 16 个常驻 warp；做完一个，队列里的下一个顶进来，一批批做完 3,907 个。GPU 有 4 个 SM 还是 400 个，**同一份代码**——grid 把程序和 GPU 规模解耦。
大白话：**你写的是订单（3,907 个 256 人的班组）；硬件决定哪些班组进哪间车间，并把每个班组编成 32 人的执行小队。**

❓ "3,907 × 256 = 1,000,192 > 一百万，多出来的 192 个线程呢？"→ 边界检查 `if (idx < N)` 的意义（Part II 细讲）。

## 第 17 页｜Let the Workshop Move（让车间动起来）⭐

🎤 先定位这一页：前面讲的是并行的**第一层收益**——很多线程同时计算；这一页讲**第二层收益**——某个 warp 暂时动不了时，其他 warp 怎么让 SM 保持忙碌。这才是 latency hiding 的正式出场。把第 9 页那张车间图**动起来**：假设某个分区此刻：Warp A 在等 HBM 数据、Warp B 在等上一条计算结果、Warp C 和 D 数据都齐了。调度员**不会**傻等 A 和 B——它直接发射就绪的 C 或 D，跳过停顿的 warp 不花任何代价。下一拍它可能又挑别的：周期 1 发 C、周期 2 发 F、周期 3 发 D、周期 4 发 H……
两个精确的说法：①"选中"更准确说是**发射（issue）指令**——指令开始执行，不代表这一拍执行完；② 延迟隐藏**不是**让某个 warp 的等待消失，而是**它等待时让别的 warp 干活**，把等待"填满"。
这就把第 5 页的 64 这个数字讲活了：64 个常驻 warp 的意义，就是给 4 个调度器准备足够深的**就绪候选池**——反过来，占用率低 = 池子浅 = 没得可切 = 等待变成纯空转。这句话记住，第 38 页起的 22 倍案例就是它的实证。
大白话（v2 收束句，只强调这一句）：**延迟隐藏不是缩短等待，而是用别的小组的活儿把等待填满。**

## 第 18 页｜Occupancy Gives the Scheduler More Choices（occupancy = 给调度员更多选择）⭐

🗣 **v2 必说（讲完定义马上补，防止第 52 页"反转"显得突然）**："occupancy 不是 GPU 有多忙，也不是性能分数。它只说明 SM 上有多少 resident warps 可供 scheduler 挑选。它提高的是调度器**找到 ready warp 的概率**——不保证一定存在 ready warp，也不保证性能一定更高。足够多可以隐藏延迟；达到足够之后，继续提高可能没有收益。"

🎤 现在正式定义标题里的概念。**Occupancy = SM 上活跃 warp 数 ÷ 硬件上限（Blackwell 是 64）**。profiler 里实测的平均值叫 **achieved occupancy**。
为什么要高？**latency hiding（延迟隐藏）**——一个 warp 卡在访存上，另一个随时顶上。Blackwell 的大寄存器堆（64K/SM）让高占用率更容易达到。
但立刻要说"制衡"：warp 们**共享寄存器和共享内存**。塞太多 warp → 每线程分到的寄存器变少 → **register spilling（寄存器溢出）**到慢速内存——你亲手制造了新的停顿。所以书里的忠告是：**把 occupancy 和寄存器/共享内存用量放在一起 profile**。
大白话：**occupancy 不是"GPU 有多忙"，更不是性能分——它只是调度员手里有多少支常驻候选队伍：够用就能藏延迟，超过够用再堆未必有收益（第 52 页会呼应这一点）。**

📚 原书精读：
> "Keeping more warps in flight is known as high occupancy on the SM... when one warp stalls, another is ready to run."

## 第 19 页｜Hardware Limits I: Warp and Block（硬件上限之一）

🗣 **v2 开场定调（对应页面顶部灰字）**："这两页的数字不需要记住。我们只需要理解一件事：每个 block 占用的 threads、registers 和 shared memory，决定一个 SM 还能同时放下多少 block 和 warp。"讲完 256 与 1024 线程 block 的例子即可，不逐行念表。

🎤 查表页，重点讲右边的框：**为什么必须是 32 的倍数**——一个 **33 线程的 block 要占两个 warp 槽位**，第二个 warp 只有 1/32 的车道干活，却照样占一个完整的调度器名额。每个"不是 32 倍数"的选择都在给虚空捐算力。
表格三行：warp 固定 32；每 block 最多 **1024 线程**（三个维度乘起来 ≤1024）；也就是每 block 最多 32 个 warp。
下面两条补充：block 太大 → 寄存器要得太多 → **溢出**；共享内存也是**SM 上全部常驻 block 共享** 227 KB。反过来，**block 小一点往往 occupancy 更高**——每 SM 能塞更多独立的 block。

## 第 20 页｜Hardware Limits II: SM-Resident and Grid（硬件上限之二）

🎤 右边图 6-9 先讲：它把线程世界的**尺度阶梯**画成一张图，从里到外一路"×32、×32、×2"：**1 个线程 → ×32 = 1 个 warp → ×32 = 单 block 上限（1,024）→ ×2 = 单 SM 常驻上限（2,048）→ grid 基本无上限（X 维约 21 亿个 block）**。这页你只需要记住这串乘法，表格都是它的展开。工厂话术：一名工人 → 32 人小队 → 一个班组最多 32 支小队 → 一间车间同时在册 64 支小队 → 订单簿想开多长开多长。
左表是每 SM 的常驻（resident）上限：**64 warp、2048 线程、32 block**——64 这个数已经**好几代不变**，所以 occupancy 的经验能跨代延续。
算一笔账（这页最有用的部分）：每 block 1024 线程 → 一个 SM 只能驻 **2 个 block**；改成 256 线程 → 能驻 **8 个**——同样 2048 线程，灵活性完全不同。
grid 上限：X 维约 21 亿个 block，Y/Z 各 65,535；每设备最多 **128 个 kernel 并发**。实践结论：**你永远先撞上 per-SM 限制**；真要超过 Y/Z 上限，用 2D/3D grid 或分多次启动（multilaunch）。

## 第 21 页｜Compatibility: PTX, SASS, and Fatbins（兼容性模型）

🎤 CUDA 生态的护城河之一：**前后向兼容**。三个概念：**SASS**——特定架构的最终机器码（sm_90 = Hopper、sm_100 = Blackwell），只带单架构 SASS 的二进制**上不了新 GPU**；**PTX**——虚拟指令集，驱动在加载时 JIT 即时编译成新架构的 SASS，这就是**前向兼容**的机制；带 f 的家族目标（如 `sm_100f`）只在同特性家族内可移植。
最佳实践（右框）：发 **fatbin（胖二进制）**——通用 PTX + 需要的家族专用 cubin，并为其他架构留 fallback。验证方法：设 `CUDA_FORCE_PTX_JIT=1` 强制走 PTX JIT——**二进制里没有 PTX 的话 kernel 启动直接失败**，逼你重新构建。
大白话（工程结论）：**性能代码不仅要跑得快，还要能跨 GPU 代际运行——所以发布时通常同时包含优化后的 SASS 和用于 forward compatibility 的 PTX。**

🗣 **讲法建议（40 秒版，不用讲上面的细节）**：开场用报错当钩子——"如果你见过 `no kernel image is available for execution on the device` 这个报错，讲的就是这页的事。"然后只说三句：① 编译产物有两种：SASS 是给某一代卡的最终机器码，PTX 是虚拟汇编、驱动加载时现场编译；② 只带 SASS 的程序换新一代卡就是上面那个报错；③ 所以发布打 fatbin：优化好的 SASS + 兜底的 PTX。家族目标、`CUDA_FORCE_PTX_JIT` 这些留在页面上当参考，口头不展开；有人追问再说。

