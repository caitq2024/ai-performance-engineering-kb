# 第六章讲稿 v2（中英对照，逐页对应 chapter06-presentation-v2.pdf，共 53 页 / 约 1 小时）

> **v2 变化**：新增 4 页自制直觉页（第 7 软硬件视角+工厂图、9 车间图、14 实例、15 延迟隐藏），且 **Part I 恢复原书小节顺序**（先 SM 内部、再 threads/warps/blocks/grids、再上限与兼容性），故页码与 v1 不一一对应；叙事上——第 2/3 页按原书引言编排（SIMT → CUDA → 内存 → roofline）、6 个标题改成结论句、比喻统一为"工厂/车间/小组"体系、occupancy 误区提前到第 16 页、每个 Part 之间加转场（本稿中标 🗣）。

> **用法**：左边开着 `chapter06-presentation-v2.pdf`（53 页版），右边看这份讲稿。每节对应一页，包含：
> - 🎤 **讲稿**——可直接照着念的中文口播稿（英文术语首次出现带中文解释）；
> - 📖 **页面英文对照**——幻灯片英文内容的翻译；
> - 📚 **原书精读**——书中关键英文原句 + 翻译（全部读完 ≈ 精读了第六章核心原文）；
> - ❓ **可能被问**——预判提问（部分页有）。
>
> **1 小时时间分配建议**：
> | 部分 | 页码 | 时长 |
> |---|---|---|
> | 开场（标题/主旨/路线图） | 1–3 | 4 分钟 |
> | Part I GPU 架构（含 4 页直觉页） | 4–20 | 16 分钟 |
> | Part II CUDA 编程 | 21–28 | 10 分钟 |
> | Part III 内存层级 | 29–37 | 11 分钟 |
> | Part IV Occupancy 实战 | 38–46 | 10 分钟 |
> | Part V 正确性与 Roofline | 47–50 | 6 分钟 |
> | 收尾（要点/结论/参考） | 51–53 | 3 分钟 |
>
> ⭐ 52 页不能平均用力。**核心重讲页**（每页约 90 秒）：7、10、13、14、15、16、21、25、29、38、39、41、43、44、48、50、52；其余页面是支撑这些核心页的证据，30–40 秒带过、不逐条念。赶时间时第 5、20、26、28、34 页各压到 30 秒；自制直觉页（7、9、14–15）赶时间可只挑车间图（9）和实例页（14）两页讲。
> 🗣 **v2 讲法提示**：每页底部浅蓝色的 Plain English 条是**收束句**——技术内容讲完后用它总结，别一上来就念；一页只强调其中一个关键词。

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

---

# Part I：GPU 架构（第 4–20 页）

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

## 第 6 页｜SFUs and Load/Store Pipelines（特殊功能单元与访存管线）

🎤 SM 里还有两类容易被忽略的部件。左边：**SFU（Special Function Unit，特殊功能单元）**，专算超越函数——sin、cos、倒数、开方。关键点：它有**自己独立的管线**，不占核心"数学+访存"管线的发射名额（"双发射"机制第 8 页细讲）——慢速复杂运算永远不会堵住核心管线，混合运算的 kernel 因此有更多指令级并行。
右边：**LD/ST（load/store）访存管线**，每 SM 共 16 条（每调度器 4 条），负责读写 L1/共享内存、L2 和全局显存。书里特别警告：**具体管线数量和配对规则不受保证**——判断 kernel 是"访存发射受限"还是"计算发射受限"要靠 profiling，细节查 Blackwell tuning guide。
大白话：**SFU 是商店后面的专柜——复杂业务去那儿办，快速通道保持流动。**

## 第 7 页｜Two Views of One Machine: Software vs. Hardware（同一台机器的两套视角）⭐

🎤 车间内部看得差不多了，拉远一步看全厂（右图）。这页是全章的"地图"，把**同一台机器的两套词汇**先立起来：
**软件视角（你写代码时用的）**：thread / block / grid——这是 CUDA 编程模型，你在 kernel 里只决定 grid 和 block 的大小。
**硬件视角（机器实际运行的）**：SM、warp、调度器——一个 block 整个生命周期都待在**同一个 SM** 上（不会跨 SM 拆开）；但注意 block 和 SM **不是一一对应**：一个 SM 通常同时容纳好几个 block。硬件把每个 block 自动切成 32 线程一组的 warp 交给调度器；grid 则铺满整块 GPU。
对应到工厂比喻：grid = **这一整批订单**，block = **一个班组**（整批订单拆给很多班组，每个班组进一间车间），warp = 车间把班组每 32 人编成的**执行小队**，thread = **一名工人**。**warp 是两套语言的桥**——你写代码时感觉不到它，但性能好坏全在它身上（32 倍数、分化、占用率都是 warp 的事）。
右图顺便预告第三部分的内存工厂：寄存器 = 随身工具箱，Shared Memory = 车间公共工作台，L1 = 车间临时货架，**L2 = 全厂中央中转仓，HBM = 厂外大仓库**；一次全局内存读取依次尝试 L1 → miss 走 L2 → miss 走 HBM，数据按 HBM → L2 → L1 → 寄存器原路返回。
大白话：**软件说的是 grid 和 block；硬件回答的是 SM 和 warp——warp 就是两套词汇表的交汇点。**

❓ 可能被问："block 会不会跨 SM？"→ 不会。一个 block 从生到死都在同一个 SM 上；能跨 SM 协作的是第 12 页要讲的 thread block cluster（DSMEM）。

## 第 8 页｜Four Warp Schedulers, Dual Issue（四调度器、双发射）⭐

🎤 每个 SM 其实是**四个"迷你 SM"**：四个独立 warp scheduler（warp 调度器），各带自己的派发逻辑。两层机制：
① 每个调度器每拍发射一个 warp 的指令 → **每拍最多 4 个 warp 同时推进**；
② **dual-issue（双发射）**：同一拍里同一个 warp 可以同时发一条算术指令（INT32/FP32/Tensor Core）+ 一条访存指令（load/store）。**限制：必须来自同一个 warp，不能跨 warp 拼**。
最好情况：每拍 **4 数学 + 4 访存**指令齐飞。右边表格（Table 6-1）是每拍的上限。注意表格下的小字：书里所有指标表的数值都是**示意值**，真实 benchmark 在配套 GitHub 仓库——这是全书的统一免责声明。
大白话：**同一层楼四名调度员，每拍各派出一支小组——而且小组可以一手计算、一手取料。**

📚 原书精读：
> "You can think of the SM as four 'mini-SMs' sharing on-chip resources." → 把 SM 想成四个共享片上资源的迷你 SM。
> "Note that the dual-issue must come from the same warp—and not across warps." → 双发射必须来自同一 warp。

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

## 第 12 页｜Thread Block Clusters and DSMEM（线程块簇与分布式共享内存）

🎤 传统上不同 block 的线程不能直接协作，现代 GPU 打破了这一点：**thread block cluster（线程块簇）**——一组能**跨 SM 通信**的 block，有簇级硬件 barrier。底层是 **DSMEM（分布式共享内存）**：把参与簇的各 SM 的共享内存 bank 用**片上高速互连**连成一个池子（右图 Figure 6-5）。效果：不同 block 的线程能以**片上速度**读、写、原子更新彼此的共享缓冲——**不花全局显存带宽**。这是今天大矩阵乘、LLM 负载的关键使能技术，第 10 章细讲，今天知道它存在即可。
大白话：**相邻班组在工作台之间的墙上开了个门，零件直接递过去，不用再走仓库。**

📚 原书精读：
> "This unification allows threads in different blocks to read, write, and atomically update one another's shared buffers at on-chip speeds—and without using global memory bandwidth."

## 第 13 页｜Warps and SIMT: 32 Threads in Lockstep

🎤 上一页的三层是**软件视角**；硬件看到的还有一层：block 会被再切成 **warp，固定 32 个线程一束**，在 **SIMT**（single instruction, multiple threads，单指令多线程）模型下**锁步（lockstep）执行**——32 个人同一拍做同一个动作。三个要点：① **硬件真正调度的单位是 warp，不是单个线程**——这是理解 GPU 的关键一跳；② warp 的大小**每一代 GPU 都是 32**，这个数可以焊死在脑子里；③ 硬件靠**快速切换 warp** 来隐藏长延迟事件（全局加载、缓存填充、管线停顿）。
大白话：**warp 是 32 人的施工小组——同一条指令全组一起动，否则整组等着。**
有了 warp 这个词，我们就可以打开 SM 看内部了——接下来三页全在 SM 里面。

## 第 14 页｜Worked Example: One Million Adds Through Both Views（实例：一百万次加法的双视角）⭐

🎤 前面这些概念，用一个具体例子全部串起来。程序员定义一个一百万元素的加法，启动方式是 `add<<<3907, 256>>>(A, B, C, N);`——左栏是**软件视角，你写的订单**：这次 launch 就是 1 个 grid；grid 里有 **3,907 个 block**（一百万除以 256 向上取整）；每个 block **256 个线程**——到这里为止都是**你自己选的**。再往下一行是硬件接手：每个 block 被切成 256÷32 = **8 个 warp**；全场约一百万线程、31,256 个 warp。每个线程只需要说一句话："我处理第 idx 个元素。"
右栏是**硬件视角**。关键点：**这 3,907 个 block 不会同时进 GPU**。为了好算，假设一块玩具 GPU 只有 4 个 SM、每个 SM 一次放 2 个 block：CUDA 先把 8 个 block 放进 4 个车间；每个 SM 此刻有 2×8 = **16 个常驻 warp**；调度器不断从里面挑**就绪的** warp 发射（这就是延迟隐藏——下一页逐拍展开）；某个 block 干完了，队列里 3,899 个还在等的 block 就顶进来一个——**一批一批地做，直到 3,907 个全部完成**。
🗣 顺势回答"为什么要这么多层"：**Thread** 让你只描述"一个元素怎么处理"，不用手写一百万次；**Block** 是协作边界——一百万个线程不可能共享同一块快速内存，所以协作（shared memory、`__syncthreads()`）被限制在 block 内部，block 之间相互独立——正因为独立，CUDA 才敢把它们随便撒到不同 SM；**Warp** 是硬件的省钱手段——32 个线程共用一条指令，取指/译码/调度成本除以 32（SIMT 的意义）；**Grid** 让程序和 GPU 规模解耦——你只说"我有 3,907 个 block 的活"，GPU 有 4 个 SM 还是 400 个 SM，由 runtime 分批调度，**同一份 kernel 不用改**。
两个容易讲错的地方：① 一个 block 只会在一个 SM 上执行，**不会跨 SM 拆开**；② block 和 SM **不是一一对应**——一个 SM 通常同时容纳多个 block。
大白话：**你写的是订单（3,907 个 256 人的班组）；硬件决定哪些班组进哪间车间，并把每个班组编成 32 人的执行小队。**

❓ 可能被问："3,907 × 256 = 1,000,192 > 一百万，多出来的 192 个线程呢？"→ 这正是第 24 页边界检查 `if (idx < N)` 存在的原因：最后一个 block 里多出来的线程直接返回。
❓ "每个 SM 到底能放几个 block？"→ 由该 kernel 的寄存器/共享内存用量和硬件上限共同决定（第 18–19 页的账），"2 个"只是为了好算的假设。

## 第 15 页｜Let the Workshop Move（让车间动起来）⭐

🎤 先定位这一页：前面讲的是并行的**第一层收益**——很多线程同时计算；这一页讲**第二层收益**——某个 warp 暂时动不了时，其他 warp 怎么让 SM 保持忙碌。这才是 latency hiding 的正式出场。把第 9 页那张车间图**动起来**：假设某个分区此刻：Warp A 在等 HBM 数据、Warp B 在等上一条计算结果、Warp C 和 D 数据都齐了。调度员**不会**傻等 A 和 B——它直接发射就绪的 C 或 D，跳过停顿的 warp 不花任何代价。下一拍它可能又挑别的：周期 1 发 C、周期 2 发 F、周期 3 发 D、周期 4 发 H……
两个精确的说法：①"选中"更准确说是**发射（issue）指令**——指令开始执行，不代表这一拍执行完；② 延迟隐藏**不是**让某个 warp 的等待消失，而是**它等待时让别的 warp 干活**，把等待"填满"。
这就把第 5 页的 64 这个数字讲活了：64 个常驻 warp 的意义，就是给 4 个调度器准备足够深的**就绪候选池**——反过来，占用率低 = 池子浅 = 没得可切 = 等待变成纯空转。这句话记住，第 38 页起的 22 倍案例就是它的实证。
大白话（v2 收束句，只强调这一句）：**延迟隐藏不是缩短等待，而是用别的小组的活儿把等待填满。**

## 第 16 页｜Occupancy Gives the Scheduler More Choices（occupancy = 给调度员更多选择）⭐

🗣 **v2 必说（讲完定义马上补，防止第 52 页"反转"显得突然）**："occupancy 不是 GPU 有多忙，也不是性能分数。它只说明 SM 上有多少 resident warps 可供 scheduler 挑选。它提高的是调度器**找到 ready warp 的概率**——不保证一定存在 ready warp，也不保证性能一定更高。足够多可以隐藏延迟；达到足够之后，继续提高可能没有收益。"

🎤 现在正式定义标题里的概念。**Occupancy = SM 上活跃 warp 数 ÷ 硬件上限（Blackwell 是 64）**。profiler 里实测的平均值叫 **achieved occupancy**。
为什么要高？**latency hiding（延迟隐藏）**——一个 warp 卡在访存上，另一个随时顶上。Blackwell 的大寄存器堆（64K/SM）让高占用率更容易达到。
但立刻要说"制衡"：warp 们**共享寄存器和共享内存**。塞太多 warp → 每线程分到的寄存器变少 → **register spilling（寄存器溢出）**到慢速内存——你亲手制造了新的停顿。所以书里的忠告是：**把 occupancy 和寄存器/共享内存用量放在一起 profile**。
大白话：**occupancy 不是"GPU 有多忙"，更不是性能分——它只是调度员手里有多少支常驻候选队伍：够用就能藏延迟，超过够用再堆未必有收益（第 52 页会呼应这一点）。**

📚 原书精读：
> "Keeping more warps in flight is known as high occupancy on the SM... when one warp stalls, another is ready to run."

## 第 17 页｜Warp Divergence（warp 分叉）⭐

🎤 SIMT 的"齐步走"有个天生软肋：分支。**同一个 warp 内**如果有人走 `if`、有人走 `else`，硬件只能**串行化**：先蒙住（mask）走 else 的那些"车道（lane）"执行 if 路径，再反过来执行 else 路径（右图 Figure 6-8）。执行时间**乘以分支路径数**。
两个要点：① **跨 warp 无惩罚**——不同 warp 各走各的分支完全免费；② 实用推论：分支条件尽量**按 warp 对齐**——按 `threadIdx.x / 32` 分支无害，按 `threadIdx.x % 2` 分支是最坏情况（每个 warp 都劈成两半）。检测和治理在第 8 章。
大白话：**同一小组一半人接了 A 工单、一半接了 B 工单，全组就得把两份活各干一遍。**

📚 原书精读：
> "warp divergence multiplies the overall execution time by the number of branches."
> "Divergence is an issue only for threads within a single warp."

## 第 18 页｜Hardware Limits I: Warp and Block（硬件上限之一）

🗣 **v2 开场定调（对应页面顶部灰字）**："这两页的数字不需要记住。我们只需要理解一件事：每个 block 占用的 threads、registers 和 shared memory，决定一个 SM 还能同时放下多少 block 和 warp。"讲完 256 与 1024 线程 block 的例子即可，不逐行念表。

🎤 查表页，重点讲右边的框：**为什么必须是 32 的倍数**——一个 **33 线程的 block 要占两个 warp 槽位**，第二个 warp 只有 1/32 的车道干活，却照样占一个完整的调度器名额。每个"不是 32 倍数"的选择都在给虚空捐算力。
表格三行：warp 固定 32；每 block 最多 **1024 线程**（三个维度乘起来 ≤1024）；也就是每 block 最多 32 个 warp。
下面两条补充：block 太大 → 寄存器要得太多 → **溢出**；共享内存也是**SM 上全部常驻 block 共享** 227 KB。反过来，**block 小一点往往 occupancy 更高**——每 SM 能塞更多独立的 block。

## 第 19 页｜Hardware Limits II: SM-Resident and Grid（硬件上限之二）

🎤 每 SM 的常驻（resident）上限：**64 warp、2048 线程、32 block**——64 这个数已经**好几代不变**，所以 occupancy 的经验能跨代延续。
算一笔账（这页最有用的部分）：每 block 1024 线程 → 一个 SM 只能驻 **2 个 block**；改成 256 线程 → 能驻 **8 个**——同样 2048 线程，灵活性完全不同。
grid 上限：X 维约 21 亿个 block，Y/Z 各 65,535；每设备最多 **128 个 kernel 并发**。实践结论：**你永远先撞上 per-SM 限制**；真要超过 Y/Z 上限，用 2D/3D grid 或分多次启动（multilaunch）。

## 第 20 页｜Compatibility: PTX, SASS, and Fatbins（兼容性模型）

🎤 CUDA 生态的护城河之一：**前后向兼容**。三个概念：**SASS**——特定架构的最终机器码（sm_90 = Hopper、sm_100 = Blackwell），只带单架构 SASS 的二进制**上不了新 GPU**；**PTX**——虚拟指令集，驱动在加载时 JIT 即时编译成新架构的 SASS，这就是**前向兼容**的机制；带 f 的家族目标（如 `sm_100f`）只在同特性家族内可移植。
最佳实践（右框）：发 **fatbin（胖二进制）**——通用 PTX + 需要的家族专用 cubin，并为其他架构留 fallback。验证方法：设 `CUDA_FORCE_PTX_JIT=1` 强制走 PTX JIT——**二进制里没有 PTX 的话 kernel 启动直接失败**，逼你重新构建。
大白话（工程结论，40 秒带过）：**性能代码不仅要跑得快，还要能跨 GPU 代际运行——所以发布时通常同时包含优化后的 SASS 和用于 forward compatibility 的 PTX。**

---

# Part II：CUDA 编程（第 21–28 页）

## 第 21 页｜Anatomy of a CUDA Kernel: Device Side（kernel 解剖：设备侧）⭐

🗣 **转场 Part I→II（对应页面顶部灰色转场行）**："刚才从硬件角度看，GPU 需要很多 ready warps。接下来从程序员角度看：CUDA 代码究竟怎样创造这些 warps？"

🎤 全书第一段完整 CUDA 代码的设备侧。四个零件：
① **`__global__`**：跑在 device、从 host 调用；
② 三个内置变量拼出**全局唯一编号**：`idx = blockIdx.x * blockDim.x + threadIdx.x`——blockIdx 是"我在第几个班组"、blockDim 是"班组多大"、threadIdx 是"我是几号"；
③ **`if (idx < N)` 边界检查**（第 24 页专门讲为什么）；
④ host 侧的启动语法 **`<<<blocksPerGrid, threadsPerBlock>>>`**——任何 kernel 调用的两个核心参数。
最重要的思维转变：**你写的是"一个工人的作业说明书"，CUDA 复印一百万份，每份发一个不同的行号。**

## 第 22 页｜Host Side: The Six-Step Data Flow（主机侧：六步数据流）

🎤 host 侧完整代码，六步在注释里标了号：**① 分配**——注意 `cudaMallocHost` 分配的是 **pinned（页锁定）内存**，不会被操作系统换页，这是后面异步拷贝能真正重叠的前提；**② H2D 拷贝**；**③ 启动**——256 线程/block，`(N+255)/256` 向上取整（N=100 万时 = 3907 个 block）；**④ `cudaDeviceSynchronize()` 等设备完成**；**⑤ D2H 拷回**；**⑥ 清理**。
两个习惯请照抄：`h_` 前缀 = host 指针、`d_` 前缀 = device 指针——全书通用。书里注明这段**还没有优化**——它是后面全书持续改进的"简单、完整的模板"。

## 第 23 页｜Why Pass N?（为什么要传 N）

🎤 初学者常见疑问：kernel 为什么不能自己看数组多长？蓝框是书里的原文回答：**CUDA kernel 的设计就是"在单个线程内工作、与几千个线程并肩、处理输入数据的一个分区"——N 定义了分区的边界。** CPU 函数可以问容器要 size；kernel 拿到的是**裸指针**，必须被告知世界的尽头在哪。配合三个内置变量，N 让每个元素被**干净且唯一**地并行处理。
大白话：**一百万份复印的作业说明书内容相同，N 是上面写"活到哪儿为止"的那一行。**

📚 原书精读：
> "a CUDA kernel function is designed to work inside of a single thread, alongside thousands of other threads, on a partition of the input data."

## 第 24 页｜The Bounds Check — and Lazily Surfacing Errors（边界检查与懒惰报错）⭐

🎤 左边是书里 N=63 的推演：63 个元素，调度器派 **2 个 warp**（64 线程）。第一个 warp 处理 0–31 没问题；第二个 warp 的最后一个线程如果不检查就会**读越界地址** → `cudaErrorIllegalAddress`。书里的金句：CUDA kernel 里到处是边界检查，**"如果没看到，你应该弄明白它为什么不在。"**
右边讲 CUDA 报错的反直觉机制：kernel **异步执行、没有每线程异常**——非法访问只是给整次 launch 设一个**全局故障标志**，host **要等下一次同步或 CUDA API 调用**才看到——错误是**懒惰浮现**的。规范写法：启动后紧跟 `cudaGetLastError()` + `cudaDeviceSynchronize()`。
大白话：**GPU 出事不会给你打电话——它留张字条，你下次开信箱才看到。**

## 第 25 页｜Choosing Launch Parameters: The 256 Recipe（256 食谱）

🎤 为什么从 256 开始？书里给了四个理由：① **32 的倍数**——没有半空 warp 占调度器名额；② **延迟隐藏**——8 个 256 线程的 block *可以*填满 SM 的 2048 线程容量（前提是寄存器和共享内存的用量允许：2048 只是线程数上限，不是任何 kernel 都真能同时驻留 8 个 block）；③ **occupancy**——8 warp/block 通常不会把寄存器和共享内存用爆；④ **资源均衡**——离 1024 上限远、调整余地大。Blackwell 的建议区间：**256–512**。grid 公式 `(N+255)/256` 向上取整保证全覆盖。
大白话：**256 是 block 尺寸里的"中杯咖啡"——几乎不会点错，之后按 profiling 微调。**

## 第 26 页｜2D and 3D Kernel Inputs（二维与三维输入）

🎤 图像这类天然二维的数据，用 2D grid × 2D block。左边代码三处变化：坐标变成 `x` 和 `y` 两个（用 `.x` 和 `.y` 分量）；边界检查变成 `if (x < width && y < height)`；再用 `idx = y * width + x` 摊平成一维下标访问。host 侧用 `dim3` 类型——比如 1024×1024 的图配 16×16 的 block（还是 256 线程）。同一套路用 `dim3(x,y,z)` 直接推广到 3D 体数据。书里说明：全书大多用 1D 或 2D（tiled 分块）配置，1D 时用普通 int 就行。
大白话：**同一份食谱、两个坐标轴：每个工人的工牌从"行号"换成"（行，列）"。**

## 第 27 页｜Allocate Asynchronously（异步分配：流与内存池）⭐

🎤 隐藏成本警告：`cudaMalloc`/`cudaFree` 是**同步且贵**的——全设备同步 + 操作系统调用（mmap/ioctl）+ 内核态切换。训练循环里每轮分配释放，积少成多。
解法三步（左边代码）：建**非阻塞 stream**（stream = GPU 上的"传送带"，同带内按序、异带互不干扰）→ `cudaMallocAsync` 在带上分配 → `cudaFreeAsync` 释放。底层是**内存池**：释放的内存回池等复用，不找 OS 要新的——省系统调用、减少**碎片化**。关键：`cudaFreeAsync` **只等自己这条 stream**，没有全局同步。
书里还有个提示：用 `cudaStreamNonBlocking` 建流是为了避开**老式默认流的隐式全局屏障**（第 11 章展开多流重叠）。
大白话：**在停车场养一支车队、用时拿钥匙——别每次送货都买车再卖车。**

## 第 28 页｜Tuning the Pool; PyTorch's Caching Allocator（池调优与 PyTorch 分配器）

🎤 两个进阶旋钮：**`cudaMemPoolAttrReleaseThreshold`**——池子保留多少内存不还给系统；**`cudaMemPoolTrimTo`**——主动归还。权衡的是"总显存占用"和"碎片化"。
右边是跟大家日常最相关的连接：**PyTorch 的 caching allocator**（配置项 `PYTORCH_ALLOC_CONF`，旧名 PYTORCH_CUDA_ALLOC_CONF）就是同一思路——复用显存、避免每建一个 tensor 都调一次同步的 cudaMalloc。
书里的选型结论：一次性缓冲区用阻塞版没问题；**分配密集的长循环用 async + 池**，性能更稳、吞吐更高。

---

# Part III：内存层级（第 29–37 页）

## 第 29 页｜The Memory Ladder at a Glance（内存梯子总览）⭐建议讲 2 分钟

🗣 **转场 Part II→III（对应页面顶部灰色转场行）**："我们已经知道怎样产生足够多的线程。但这些线程大部分时间究竟在等什么？答案通常是数据——所以接下来要看 memory hierarchy。"

🎤 全章核心表格（Table 6-5），从上往下**容量越来越大、速度越来越慢**：寄存器（单周期、几十 TB/s）→ 共享+L1（20–30 拍、TB/s）→ TMEM（Tensor Core 专用）→ 常量缓存（1 拍广播）→ L2（**126 MB**、约 200 拍）→ local memory（寄存器溢出区，实际在 DRAM！）→ HBM3e（**180 GB、约 8 TB/s**、几百到一千拍）。
一句话行动准则：**能复用就往上层放，必须下 HBM 就合并访存。**接下来 8 页逐层拆开讲。
大白话：**随身工具箱（寄存器）→ 公共工作台（共享内存）→ 全厂中转仓（L2）→ 远处大仓库（HBM）。活儿尽量在自己工位上干。**

📚 原书精读：
> "maximizing data reuse in registers, shared memory, and L1/L2 cache—and minimizing reliance on global memory—is essential for high-throughput GPU kernels."

## 第 30 页｜Register Pressure Can Turn Fast Storage into DRAM Access（寄存器压力会把最快的存储变成 DRAM 访问）

🎤 书里的说法很形象：每个线程"从寄存器堆开始它的旅程"——单周期读写、几乎不与任何东西争抢、每 SM 几十 TB/s。预算：64K/SM、**每线程最多 255 个**。
然后是**悬崖**：需要更多（局部变量太多、编译器临时量太多）→ 溢出（spill）进 **local memory**——名字里有 local，物理上在**片外 DRAM**，几百到一千多拍。这是 CUDA 里最经典的"静默性能杀手"。监控指标：Nsight Compute 的 **Registers Per Thread**。
大白话：**"local memory" 是 CUDA 里最误导人的名字——它是城另一头的自助仓库，不是你的口袋。**

## 第 31 页｜Shared Memory + L1: One SRAM, Two Jobs（共享内存与 L1）

🎤 一块 **256 KB 的 SRAM 干两份工**：用户管理的共享内存（上限 228/227 KB）+ L1 数据缓存。分割比例（carveout）你自己选——左边代码 `cudaFuncSetAttribute(...PreferredSharedMemoryCarveout...)`。
性能：20–30 拍；避开 **bank conflict（存储体冲突）**——多个线程撞到同一个 bank 会串行化——就能拿到 TB/s 级吞吐。这里是**块内协作的工作台**：矩阵分块（tile）、归约、数据暂存都在这儿。
大白话：**一张工作台，隔板可调：多少归你们班组的项目台，多少归自动的缓存架。**

## 第 32 页｜TMEM and TMA: Feeding the Tensor Cores

🎤 Blackwell 新增：**TMEM**，每 SM 256 KB 专用 SRAM，是第五代 Tensor Core 指令（tcgen05、**UMMA**——统一矩阵乘累加）的**累加器**，与 Tensor Core 之间几十 TB/s。
特别之处：**CUDA C++ 里拿不到它的指针**——数据进出全由 **TMA（张量内存加速器）**按"描述符"自动编排。看右图的 C = A×B：操作数 B 在共享内存、A 和累加器在 TMEM；TMA 负责 HBM→L2→SMEM 的搬运，SMEM↔TMEM 由 Tensor Core 指令隐式完成。净效果：**大幅减少 Tensor Core 对全局显存的依赖**。细节第 10 章。
大白话：**TMA 是专职叉车队，在后台不停搬运物料，做矩阵乘的大厨永远不用离开厨房。**

## 第 33 页｜Constant Cache: One-Cycle Broadcast（常量缓存：单拍广播）

🎤 每 SM 约 8 KB 的小缓存，前置 64 KB 的 `__constant__` 只读空间。绝活：**全 warp 32 个线程读同一个地址时，1 拍广播给所有人——和读寄存器一样快**。反之，分歧读会跨车道串行化。所以它适合**小、只读、所有线程访问同一地址**的数据。
右边绿框是书里点名的 LLM 场景（都是高频小表）：**RoPE 旋转位置编码查找表、ALiBi 斜率、LayerNorm 的 γ/β 向量、embedding 量化 scale**——全体线程共享、零全局显存流量。
大白话：**它是喇叭不是信箱：一次广播全班 32 人都听到——前提是大家问的是同一个问题。**

## 第 34 页｜L2 Cache: The GPU-Wide Middleman（L2：全 GPU 的中间人）

🎤 **126 MB**、所有 SM 共享，是片上 SRAM 和片外 HBM 之间的胶水。约 200 拍、聚合带宽几十 TB/s，吸收 L1 溢出。最有价值的性质：**跨 block 复用**——一个 block 取过的数据，其他 block 从 L2 拿，不用重访 DRAM。
右框是书里的合并访存法则：**把全局加载组织成 128 字节对齐的 coalesced（合并）段**，干净映射到缓存线——避免事务被拆分，同时拉满 L2 和 DRAM 带宽。具体手法第 7 章。
大白话：**L2 是全厂的中转仓：别的小组已经从大仓库运来的零件，你直接到中转仓取——前提是大家按"整托盘"下单，别一颗螺丝一颗螺丝地要。**

## 第 35 页｜Global HBM3e, the Dual-Die B200, and Coherency

🎤 梯子最底层：**HBM3e**，B200 是 180 GB（B300 约 288 GB）、约 8 TB/s——容量最大、带宽惊人，但**延迟几百到一千多拍，是链条上最慢的一环**。寄存器溢出、超大自动数组（local memory）也在这里付同样的价。
两个冷知识：① 书里的边栏——**B200 物理上是两块 die**（受光刻极限限制），10 TB/s 片间互连、各接 4 个 HBM 栈，但呈现为**一个 GPU、一个地址空间**；② **point of coherency（一致性生效点，Figure 6-15）**：内存一致性按 thread → block → cluster → device → system 五级建立——**通信范围越广，代价越高**。

## 第 36 页｜Unified Memory Hides Copies, Not Their Cost（统一内存隐藏的是拷贝，不是拷贝的代价）

🎤 `cudaMallocManaged()`：CPU+GPU 一个一致的地址空间——不用分开管缓冲、不用手写 memcpy。底层机制：页**按需迁移**。
"catch"（代价）在这里：GPU 摸到一个还在 CPU 内存的页 → **缺页（page fault）、kernel 停等**。硬件差异巨大：PCIe 上按缺页搬运**可能比手动 memcpy 还慢**；Grace 超级芯片的 **NVLink-C2C 约 900 GB/s**，迁移接近原生速度——**但延迟永远不是零**。
大白话：**统一内存是自动送料服务——省事，但小组要用时零件还没到车间，全组就只能站着等。**

📚 原书精读：
> "any unexpected page-fault during a kernel launch will stall the GPU while the runtime moves the needed page into place."

## 第 37 页｜Taming Unified Memory: Prefetch, Advise, Attach（驯服统一内存）

🎤 治理"意外缺页"的三板斧（左边代码从上到下）：
① **`cudaMemPrefetchAsync`**——kernel 启动**前**整体预取，把"首次触碰迁移"变成可重叠的异步传输；
② **`cudaMemAdvise` 三条建议**：`SetPreferredLocation`（数据主要在哪用）、`SetReadMostly`（基本只读，驱动可以两边各放副本）、`SetAccessedBy`（让另一块 GPU 直接映射、不触发迁移）；
③ **`cudaStreamAttachMemAsync`**——把一段内存绑定到一条 stream，别的 stream 不再因它意外停顿。
补充：没有 NVLink-C2C 的多卡系统，用 peer copy/预取把数据钉在 NUMA 本地。书里的结论：三板斧用齐，统一内存性能**非常接近手动 cudaMemcpy**，同时保住简洁性。

---

# Part IV：Occupancy 实战（第 38–46 页）

## 第 38 页｜Occupancy Ground Rules（占用率基本法则）⭐

🗣 **转场 Part III→IV（对应页面顶部灰色转场行）**："内存层级告诉我们 warp 为什么会 stall。现在回到 occupancy：如果一个 warp 在等数据，我们究竟需要多少其他 warp 才能填满这些空档？"

🎤 蓝框是书里"CUDA 性能最基本的法则"原文：**"Launch enough parallel work to fully occupy the GPU."**——启动足够多的并行工作填满 GPU。
下面两条规则务必分清：**规则一**——occupancy 低且性能差：第一味药是**加并行度**，加到"有足够多的 ready warp 能把延迟藏住"为止——**性能不再上涨（plateau）或别的瓶颈占主导时就停**，不要以某个固定百分比为目标（第 43 页 38.7% 拿到 22 倍就是证据）；**规则二**——occupancy 已经中高但 kernel 是 memory-bound：推到 100% **没用**——你只需要"刚好够藏延迟"的 warp 数，之后瓶颈在带宽。
接下来 5 页是一个完整案例：同一个操作（C = A + B，一百万元素）、两种实现、profiler 的裁决。
大白话：**规则一是让车间站满小组；规则二提醒你：车间站满了，卡在仓库运货卡车上照样停工。**

## 第 39 页｜Case Study, Take 1: addSequential（案例上：串行版）

🎤 反面教材：只有 `blockIdx.x==0 && threadIdx.x==0` 的**一个线程**for 循环加完一百万个元素，启动配置 `<<<1,1>>>`（书里注明：这个 kernel 的语义就是单线程，别改启动配置）。
后果（书中原话）："GPU 的庞大资源基本闲置……只有一个 warp、甚至只有其中一个线程在干活。"更致命的是：**没有其他 warp 可切换 → 每次访存等待都是纯粹的空转 → 延迟隐藏为零**。
大白话：**你租下整座工厂，雇了一个工人。**

## 第 40 页｜The Same Trap in PyTorch（PyTorch 里的同款陷阱）⭐

🎤 同样的错误在 PyTorch 里更常见也更隐蔽——左边代码：用 Python for 循环逐元素 `C[i] = A[i] + B[i]`。这会**串行发射一百万个迷你 kernel**，书里的说法是把 GPU 用成了"标量、非并行处理器"。occupancy 跟 addSequential 一样惨，还外加**一百万次 kernel 启动的 CPU 开销**。
书里的忠告：除非你在写全新的东西，**几乎总有现成的 PyTorch 原生向量化实现**——包括 PyTorch 编译器生成的代码。**不要在 GPU 操作外面套 Python 循环。**
大白话：**GPU 上最常见的性能 bug，是不小心把 GPU 当成了一台很贵的单核 CPU。**

## 第 41 页｜Case Study, Take 2: addParallel — and C = A + B（案例下：并行版）

🎤 正确姿势：每线程加**一个**元素，`<<<(N+255)/256, 256>>>` 启动约 3907 个 block、一百万个线程（Figure 6-19）。代码里两个细节：**`__restrict__`** 注解承诺指针无别名（aliasing），解放编译器优化；书中完整版还用了 Part II 教的全套习惯——pinned 内存、非阻塞 stream、cudaMallocAsync/cudaMemcpyAsync 全异步链路。
PyTorch 版就一行：**`C = A + B`**——单个向量化 kernel，海量线程并行。
大白话：**同一座工厂，现在每个工位都有人——PyTorch 版更省事：直接找劳务派遣公司。**

## 第 42 页｜Measuring It: nsys and ncu（量化验证的命令）

🎤 书里给了完整命令行（左边），分工记一句话：**nsys 回答"时间去哪了——GPU 是被饿着还是被堵着"；ncu 回答"这个 kernel 为什么慢——occupancy？停顿？缓存？"**。必须**两个都跑**：只跑 nsys 看不到 kernel 内部低效；只跑 ncu 不知道 kernel 是否被及时喂数据。
一个彩蛋：ncu 命令里那个很长的指标名 `sm__warps_active.avg.pct_of_peak_sustained_active` **就是 achieved occupancy 本尊**——面试可以拿出来说。
大白话：**nsys 是整台机器的秒表，ncu 是对准一个 kernel 的显微镜。**

## 第 43 页｜Parallelism Cuts Runtime 22× at Only 38.7% Occupancy（22 倍提速，只用了 38.7% 占用率）⭐

🎤 裁决（Table 6-6，示意值）：**kernel 时间 48.21 → 2.17 ms，22 倍**；GPU 利用率 1.5% → 95%；achieved occupancy 1.3% → **38.7%**；warp 执行效率 3.1% → 100%——3.1% 就是 1/32：一个 warp 里只有一条车道干活。
两个洞察：① **38.7% 的占用率就买到了 22 倍**——根本不需要 100%；② 右图（Figure 6-20）是加速的本质：串行版时间线是"操作+访存空档"交替；并行版**空档被其他 warp 的工作填满**——延迟隐藏的可视化。
补充：这页正好教大家分辨四个**相关但不同**的指标——**GPU utilization**（GPU 有活干的时间占比，各工具定义略有差异）、**SM Active**（SM 活跃的周期占比）、**achieved occupancy**（平均活跃 warp 数 ÷ 硬件上限）、**warp execution efficiency**（warp 内有效车道的占比）。95% utilization、38.7% occupancy、100% warp efficiency 能同时出现，正说明它们量的是**不同维度**。

📚 原书精读：
> "No matter how fast each thread is, you need lots of threads to leverage the GPU's throughput potential."

## 第 44 页｜A Busy GPU Can Still Be Waiting on Memory（忙碌的 GPU 仍可能在等内存）⭐

🗣 **全场最重要的转场（对应页面顶部灰色转场行）**："并行版本达到 95% 的 GPU utilization，却只有 38.7% 的 occupancy，而且已经快了 22 倍——这说明我们不需要 100% occupancy。但它是否已经到达 GPU 算力峰值？仍然没有，因为 vector add 主要在搬数据。"

🎤 泼冷水的一页，也是通往 roofline 的桥。占用率之后的下一层是每 warp 的效率（ILP，第 8 章）——**但即使 100% occupancy，memory-bound（受限于数据搬运）的 kernel 照样受损**。
书里的典型例子：**LLM 的 decode 阶段**——每生成一个 token 都要把**模型权重**从 HBM 流进寄存器/共享内存。几千亿参数 × 约 1 字节 ≈ **几百 GB 一遍**——不管开多少线程，显存带宽先饱和。
下面的框是书里的趋势判断：**GPU FLOPS 的增速正在甩开显存带宽**（HBM3e 约 8 TB/s，但算力和模型规模长得更快）——优化数据搬运在现代 AI 负载里绝对关键。
大白话：**当整个活儿就是"从仓库搬箱子"，在办公桌前多雇职员没有用——这正是 Part V 的 roofline 模型要形式化的事。**

## 第 45 页｜__launch_bounds__: Compile-Time Occupancy Control

🎤 第一件调优工具，编译期的。`__launch_bounds__(256, 16)` 给编译器两个信息：**承诺** block 不超 256 线程；**请求**每 SM 至少驻 16 个这样的 block。编译器于是**压缩每线程寄存器、克制展开和内联**，好塞进更多 warp。
有意思的细节：16×256 = 4096 **超过了** 2048 的硬件上限 → 编译器**削到 8 个 block** 并给出 ptxas 警告（".minnctapersm will be ignored"）——说明这些参数是愿望，硬件上限说了算。
交易的本质：**牺牲一点单线程性能，换更多 warp 在飞、更稳定的 warp 吞吐**。危险区：压得太狠 → 寄存器不够 → **溢出到 local memory**——比不优化还慢。
大白话：**拿"每个工人的工具箱大小"换"车间里能站多少工人"——甜点只能靠实验找。**

## 第 46 页｜The Occupancy API: Runtime Autotuning

🎤 第二件工具，运行时的。`cudaOccupancyMaxPotentialBlockSize()` 根据 kernel **实际的**寄存器/共享内存消耗，自动算出 occupancy 最优的 block size。
书里点名的两个坑：① 返回的 **`minGridSize` 是"喂饱 occupancy 的最小 grid"，不是覆盖 N 个数据的 grid**——真正的 grid 要取 `max(minGridSize, ceil(N/blockSize))`；② kernel 用了 `extern __shared__` 动态共享内存的话，**字节数必须如实传**。
最后是书里的边栏忠告：**API 的建议要用 ±1–2 档 block size 实测验证**——寄存器压力和 L2 行为可能让"略低于最大占用率"的配置实际更快。

---

# Part V：正确性与 Roofline（第 47–50 页）

## 第 47 页｜Compute Sanitizer（计算消毒器）

🗣 **转场 Part IV→V（对应页面顶部灰色转场行）**："所以接下来不能继续盲调 block size。我们需要一个方法判断，到底撞上了 compute ceiling 还是 memory ceiling——这就是 roofline（中间先绕一小段正确性检查）。"

🎤 换个话题：不谈快慢，谈对错。几万个线程的程序，传统 debugger 抓不住偶发的内存错误和竞争。CUDA Toolkit 自带的 **Compute Sanitizer** 运行时插桩，四件套：
左栏（内存类）：**memcheck**——越界/未对齐/泄漏（最常用）；**initcheck**——读了未初始化的显存（典型病因：**忘了 H2D 拷贝**）。
右栏（并发类）：**racecheck**——共享内存竞争（WAW/WAR/RAW 三种冒险）；**synccheck**——非法同步、错配 barrier → 死锁。
用法：`compute-sanitizer --tool <名> ./app`，`--kernel-name` 过滤、NVTX 标注。书里的最佳实践：**进 CI + `--error-exitcode`**——正确性回归在合码前拦下。
大白话：**十万个线程面前，"我这儿跑过没问题"毫无意义——sanitizer 是你的安全带。**

## 第 48 页｜Roofline Tells Us Which Resource Is the Limit（roofline 告诉你极限在哪种资源上）⭐建议讲 2–3 分钟

🎤 最后一个大概念，也是全书反复用的分析框架（右图 Figure 6-21）。
横轴：**arithmetic intensity（算术强度）**= 每从 HBM 搬 1 字节做几次浮点运算（FLOPs/byte）。两条天花板：**水平的计算屋顶**（约 80 TFLOP/s FP32）和**倾斜的内存屋顶**（约 8 TB/s），交点是 **ridge point（脊点）≈ 10 FLOPs/byte**（80T ÷ 8T）。脊点**左边 = memory-bound**（饿数据），**右边 = compute-bound**（饿计算）。
现场算一遍书里的例子：C = A + B——读 8 字节、加 1 次、写 4 字节 → **1 FLOP ÷ 12 字节 ≈ 0.083**——离脊点差 100 多倍，**无可救药地 memory-bound**。这就从数学上解释了第 44 页：为什么占用率救不了它。
对到日常：书里的边栏说 LLM 两个阶段各占一边——**prefill 偏 compute-bound、decode 偏 memory-bound**（第 15–18 章展开）。
大白话：**roofline 在你动手前先问一个问题：这个 kernel 是饿计算，还是饿数据？答错方向，白干。**

## 第 49 页｜Moving Right on the Roofline: Lower Precision Pays Twice（低精度一石二鸟）

🎤 知道自己 memory-bound 了怎么办？**往右移**：每字节多干活（片上复用、算子融合），或者最直接的——**把字节变小**。
关键账目：GPU 访存以 **128 字节一个事务**为单位，能装 **32 个 FP32 = 64 个 FP16 = 128 个 FP8 = 256 个 FP4**。FP32→FP16，算术强度**立刻翻倍**；FP8 相对 FP16 再翻一倍吞吐、再省一半内存。Blackwell 原生支持 FP8/FP4 Tensor Core。
更妙的是**硬件解压**：权重以压缩形式存 HBM（甚至 4 位/2 位方案），硬件读取时**在线解压**、再转 FP16/FP32 做高精度累加——**变相扩大可用带宽**。这就是 Blackwell 跑 memory-bound 的 token 生成特别强的架构原因。
蓝框收束：**精度既是计算优化，更是带宽优化——字节减半 ⇒ 强度翻倍 ⇒ 靠近计算屋顶。**

## 第 50 页｜Profiling Workflow: Diagnose, Fix, Re-measure（诊断-修复-复测）

🎤 方法论收尾。**memory-bound 的指标签名**：ncu 里 **DRAM 利用率高 + ALU 利用率低**（warp 停在访存上），**global load efficiency** 下降说明 DRAM 请求满足得不够快；nsys 时间线上 kernel 之间出现空闲段 = GPU 在等数据。
**重叠失败的两大病因**（右图 Figure 6-22 是期望的重叠效果）：① 不想要的**默认流同步**；② **缺 pinned 内存**——没有它 `cudaMemcpyAsync` **根本无法**和 kernel 重叠，这是书里点名的常见性能问题。
修好之后的样子：空闲段消失、memory pipe utilization 爬向峰值、端到端吞吐跳升。最后一条原则贯穿全书：**每改一处，测一次。**

---

# 收尾（第 51–53 页）

## 第 51 页｜Key Takeaways（书中八条要点）

🎤 （左右两栏各念四条，每条一句话）左栏：SIMT——32 线程锁步，多 warp 在飞才藏得住延迟；层级——thread→block→grid，少用 barrier；occupancy 与上限——32 的倍数、记住每 SM 四个数（64 warp/32 block/228 KB/255 寄存器）；启动参数——256 起步、ceil 公式、按 profiling 调。
右栏：异步内存——Async API+stream+池，PyTorch 分配器同理；内存梯子——上层复用、下层合并；统一内存——prefetch+advise 消灭意外停顿；roofline——FLOPs/byte 定战场，低精度+硬件解压往右移，**TMEM+UMMA 能把 kernel 从 memory-bound 拉向 compute-bound**。

## 第 52 页｜Conclusion and What's Next

🎤 整章一句话（蓝框）：**让 GPU 忙起来（occupancy）、让数据离计算近（内存层级）、让 roofline 告诉你下一仗往哪打。**
书的结论里有个反直觉的提醒，必须带到：**占用率最大化不总是最优**——每线程有足够 ILP（指令级并行）时，中低占用率也能跑满吞吐；有时**故意少开线程、让每线程多拿寄存器**反而更快。唯一的裁判是 benchmark。
预告：第 7–8 章讲访存模式和 warp 效率、第 9 章算术强度、第 10 章 TMA 和 warp specialization。我负责的另一章——**第 12 章**——与今天正好衔接：今天解决"单个 kernel 怎么快"，那章解决"kernel 之间怎么编排、让 GPU 永远不用等 CPU"。谢谢大家，欢迎提问。

📚 原书精读（结论段关键句）：
> "However, maximizing occupancy does not guarantee best performance in every case. GPUs can often achieve very high throughput at moderate or even low occupancy if threads have sufficient instruction-level parallelism (ILP)."

## 第 53 页｜References & Further Reading

🎤 参考资料：原书第六章；NVIDIA 的 Blackwell tuning guide 和 CUDA C++ Programming Guide（所有硬件上限的权威出处）；roofline 原始论文（Williams/Waterman/Patterson，CACM 2009）；Compute Sanitizer 和 Nsight 官方文档。表格里的数字是书中示意值，真实分架构 benchmark 在书的 GitHub 仓库。

---

# 附录 A｜答辩预备（高频问题速查）

1. **"线程按什么规则分进 warp？"** 按线程编号连续切：threadIdx 0–31 第一个 warp、32–63 第二个。所以按 `threadIdx.x/32` 分支无害、按 `%2` 分支最坏。
2. **"occupancy 是不是越高越好？"** 不是。够藏延迟即可；memory-bound 时无效；ILP 充足时低占用也能满吞吐；有时少线程多寄存器更快。实测为准（书中结论原话见第 52 页）。
3. **"表里的数字准吗？"** 书中所有指标表是示意值（illustrative），真实 benchmark 在配套 GitHub 仓库——书里有统一免责声明。
4. **"为什么 227 不是 228？"** CUDA 每 block 保留 1 KB。
5. **"LD/ST 管线到底几条？"** 书里说 16 条（每调度器 4 条）但明确警告"具体数量与配对不受保证"，以 profiling 和官方文档为准。
6. **"PTX 和 SASS 什么关系？"** PTX 是虚拟指令集（可 JIT 到新架构），SASS 是特定架构机器码；发 fatbin 两者都带；`CUDA_FORCE_PTX_JIT=1` 可验证。
7. **"统一内存和 pinned 内存什么关系？"** 是两回事：pinned（cudaMallocHost）是不可换页的 host 内存，用于快速/可重叠的显式拷贝；managed（cudaMallocManaged）是自动迁移的统一地址空间。
8. **"decode 为什么 memory-bound？"** 每 token 都要把全部权重从 HBM 过一遍：几千亿参数 × 1 字节 ≈ 几百 GB，带宽先于算力饱和；所以低精度和硬件解压（每事务装更多值）在 decode 上收益最大。

# 附录 B｜术语总表（英文 → 中文 → 一句话）

| 英文 | 中文 | 一句话 |
|---|---|---|
| host / device | 主机 / 设备 | CPU 侧 / GPU 侧 |
| kernel | 核函数 | 跑在 GPU 上的函数，`__global__` 标注 |
| SM | 流式多处理器 | GPU 的"车间"，Blackwell 有一百多个 |
| thread / thread block (CTA) / grid | 线程 / 线程块 / 网格 | 工人 / 班组（≤1024）/ 全部班组 |
| warp | 线程束 | 32 线程的最小调度单位，齐步走 |
| SIMT | 单指令多线程 | 一条指令驱动 32 个线程 |
| warp scheduler | warp 调度器 | 每 SM 四个，各自每拍发射一个 warp |
| dual-issue | 双发射 | 同 warp 同拍发 1 算术 + 1 访存 |
| SFU | 特殊功能单元 | sin/cos/sqrt 专用管线，不占双发射名额 |
| LD/ST pipeline | 访存管线 | 每 SM 16 条，读写各级内存 |
| occupancy / achieved occupancy | 占用率 / 实际占用率 | 活跃 warp ÷ 上限 64 / profiler 实测值 |
| latency hiding | 延迟隐藏 | 等内存时切换别的 warp |
| warp divergence | warp 分叉 | 同 warp 内走不同分支 → 串行化 |
| lane / mask | 车道 / 屏蔽 | warp 里的一个线程位 / 分叉时关掉不走该路的车道 |
| coalesced access | 合并访存 | 128 字节对齐连续访问，一次事务搞定 |
| register spilling | 寄存器溢出 | 寄存器不够、数据被挤到慢速 local memory |
| local memory | 局部内存 | 名字欺骗性强：物理上在 DRAM，很慢 |
| shared memory (SMEM) | 共享内存 | block 内共享的片上 SRAM |
| carveout | 划分比例 | 共享内存 vs L1 的分割，可编程设置 |
| bank conflict | 存储体冲突 | 多线程撞同一 bank，串行化 |
| constant memory | 常量内存 | 64 KB 只读区 + 8 KB 缓存，同地址读可广播 |
| TMEM / TMA | 张量内存 / 张量内存加速器 | Tensor Core 的累加器 / 按描述符自动搬运数据 |
| UMMA / tcgen05 | 统一矩阵乘累加 / 第五代 TC 指令 | Blackwell Tensor Core 指令家族 |
| HBM3e | 高带宽显存 | B200：180 GB、约 8 TB/s |
| Unified Memory / managed memory | 统一内存 / 托管内存 | CPU+GPU 一个地址空间，页自动迁移 |
| page fault / migration | 缺页 / 迁移 | GPU 摸到不在本地的页 → 停下来等搬运 |
| pinned memory | 页锁定内存 | 不被 OS 换页的 host 内存，异步拷贝重叠的前提 |
| stream | 流 | GPU 上的操作队列，"传送带" |
| memory pool | 内存池 | 复用已释放显存，避免 OS 调用 |
| `__launch_bounds__` | 启动边界注解 | 编译期承诺 block 上限、请求驻留数 |
| `__restrict__` | 无别名注解 | 承诺指针不互相重叠，解放编译器 |
| arithmetic intensity | 算术强度 | FLOPs ÷ 搬运字节数 |
| roofline / ridge point | 屋顶线 / 脊点 | 两条性能天花板 / 分界（Blackwell ≈ 10 FLOPs/B）|
| memory-bound / compute-bound | 访存受限 / 算力受限 | 饿数据 / 饿计算 |
| ILP | 指令级并行 | 单线程内多指令并行，低占用率的补偿手段 |
| NVTX | NVIDIA 工具扩展 | 给代码打时间线标签 |
| Nsight Systems / Compute | — | 整机秒表 / 单 kernel 显微镜 |
| Compute Sanitizer | 计算消毒器 | memcheck/racecheck/initcheck/synccheck |
| fatbin / PTX / SASS | 胖二进制 / 虚拟指令集 / 机器码 | 兼容性三件套：PTX 保前向兼容 |
| prefill / decode | 预填充 / 解码 | LLM 读题（偏算力受限）/ 逐 token 生成（偏访存受限）|
