# 第六章讲稿（中英对照，逐页对应 chapter06-presentation.pdf，共 52 页 / 约 1 小时）

> **用法**：左边开着 `chapter06-presentation.pdf`（52 页版），右边看这份讲稿。每节对应一页，包含：
> - 🎤 **讲稿**——可直接照着念的中文口播稿（英文术语首次出现带中文解释）；
> - 📖 **页面英文对照**——幻灯片英文内容的翻译；
> - 📚 **原书精读**——书中关键英文原句 + 翻译（全部读完 ≈ 精读了第六章核心原文）；
> - ❓ **可能被问**——预判提问（部分页有）。
>
> **1 小时时间分配建议**：
> | 部分 | 页码 | 时长 |
> |---|---|---|
> | 开场（标题/主旨/路线图） | 1–3 | 4 分钟 |
> | Part I GPU 架构（含 3 页比喻图） | 4–19 | 18 分钟 |
> | Part II CUDA 编程 | 20–27 | 10 分钟 |
> | Part III 内存层级 | 28–36 | 11 分钟 |
> | Part IV Occupancy 实战 | 37–45 | 10 分钟 |
> | Part V 正确性与 Roofline | 46–49 | 5 分钟 |
> | 收尾（要点/结论/参考） | 50–52 | 2 分钟 |
>
> ⭐ 标注的是重点页，可以多讲 30–60 秒；赶时间时第 7、19、25、27、33 页可以各压缩到 30 秒；比喻三连页（8–10）赶时间可并成一页快讲。

---

## 第 1 页｜标题页

🎤 大家好，今天我讲第六章：GPU Architecture, CUDA Programming, and Maximizing Occupancy——GPU 架构、CUDA 编程与最大化"占用率"。前五章讲的是系统层面（硬件、网络、存储），从这章起全书进入 GPU 内部。这一章是第 7 到 12 章所有 kernel 级优化的**地基**，所以今天概念比较多，我会全程用大白话解释，讲一个小时左右。

📖 核心术语先混脸熟：**Occupancy**（占用率）= GPU 的"工位坐满率"；**SM**（Streaming Multiprocessor，流式多处理器）= GPU 的"车间"；**Warp**（线程束）= 32 个线程绑在一起执行的最小调度单位；**Kernel**（核函数）= 跑在 GPU 上的函数。

---

## 第 2 页｜What This Talk Is About — and Why（讲什么、为什么）

🎤 一句大白话概括整章：**GPU 是一台"吞吐量机器"**——不追求单线程快，而是同时跑几千个线程，用"总有人在干活"来掩盖内存慢。本章教三样东西：**词汇表**（warp、block、grid、SM）；**内存的梯子**（寄存器→共享内存→L2→HBM，一层比一层大、也一层比一层慢）；和**一条黄金法则**：启动足够多的并行工作，把每个 SM 喂饱。
为什么重要？SM 空转的 GPU 就是一台昂贵的电暖器——很多没调优的 kernel 只用到硬件的百分之几。**occupancy** 是"喂饱程度"的度量，**roofline**（屋顶线）模型是指南针——在你动手之前告诉你该优化计算还是访存，避免优化错方向。

📖 页面对照：
- "A GPU is a **throughput machine**..." → GPU 是吞吐量机器：同时跑几千个线程，靠"手里永远有别的活"掩盖内存慢。
- "a GPU with idle SMs is an expensive space heater" → SM 空转的 GPU 是台昂贵的取暖器。

📚 原书精读：
> "Unlike CPUs, which optimize for low-latency single-thread performance, GPUs are throughput-optimized processors built to run thousands of threads in parallel."
> 与优化单线程低延迟的 CPU 不同，GPU 是为并行运行几千个线程而生的吞吐量优化型处理器。

---

## 第 3 页｜Roadmap: Five Parts, One Golden Rule（路线图）

🎤 今天分五个部分，页码都标在括号里方便大家跟进：**Part I GPU 架构**（4–19 页）——先讲线程怎么组织（thread/block/grid/warp），再打开 SM 看内部（我自己画了两张比喻图帮大家建立直觉），然后是 warp 分叉、硬件上限、兼容性；**Part II CUDA 编程**（20–27）——kernel 骨架、启动参数、2D/3D、异步分配；**Part III 内存层级**（28–36）——从寄存器到 HBM 的完整梯子加统一内存；**Part IV Occupancy 实战**（37–45）——一个 22 倍加速的案例和两件调优工具；**Part V 正确性与 Roofline**（46–49）。
主线一句话：**先让 GPU 忙起来，再抠每个周期，永远用 profiler 决定用哪个药方。**

---

# Part I：GPU 架构（第 4–19 页）

## 第 4 页｜GPUs Optimize Throughput; CPUs Optimize Latency

🎤 CPU：核少、缓存深、单线程快。GPU：一百多个 SM 并行跑几千个线程。右图（Figure 6-1）是最基本的协作流程：host（CPU 侧）把数据拷到 GPU 显存、启动 kernel、拷回结果——注意有两次跨设备拷贝。GPU 的应对不是把拷贝变快，而是**用海量并行把延迟藏起来**。它的甜区是**数据并行**工作：矩阵乘、卷积——同一条指令作用于海量元素；可以直接写 CUDA C++，也可以经由 PyTorch 或 OpenAI Triton 这类 Python 系工具间接生成。
大白话：**CPU 是跑车，GPU 是货运列车——要让列车装满，而不是要它快。**

📚 原书精读：
> "GPUs rely on massive parallelism to hide data-transfer latency." → GPU 依靠海量并行隐藏数据传输延迟。
> "Each GPU comprises many SMs, which are roughly analogous to CPU cores but streamlined for parallelism." → 每块 GPU 有许多 SM——粗略类比 CPU 核，但为并行精简。

## 第 5 页｜The Thread Hierarchy: Threads → Blocks → Grids（线程层级）

🎤 CUDA 把并行工作组织成三层（右图 Figure 6-3）：**thread（线程）**——处理一个数据元素的工人；**thread block（线程块，又名 CTA，协作线程阵列）**——最多 1024 线程一组，组内共享快速的片上共享内存；**grid（网格）**——一次启动的全部 block，尺寸设对可以扩展到几百万线程、kernel 一行不改。调度和分发由 CUDA 运行时（以及 PyTorch）自动完成。
大白话：**工人组成班组，班组内交流便宜；公司按活儿多少雇任意多个班组。**

📚 原书精读：
> "By sizing your grid appropriately, you can scale to millions of threads without changing your kernel logic."
> grid 尺寸设对，可扩展到几百万线程而不改 kernel 逻辑。

## 第 6 页｜Warps and SIMT: 32 Threads in Lockstep

🎤 上一页的三层是**软件视角**；硬件看到的还有一层：block 会被再切成 **warp，固定 32 个线程一束**，在 **SIMT**（single instruction, multiple threads，单指令多线程）模型下**锁步（lockstep）执行**——32 个人同一拍做同一个动作。三个要点：① **硬件真正调度的单位是 warp，不是单个线程**——这是理解 GPU 的关键一跳；② warp 的大小**每一代 GPU 都是 32**，这个数可以焊死在脑子里；③ 硬件靠**快速切换 warp** 来隐藏长延迟事件（全局加载、缓存填充、管线停顿）。
大白话：**warp 是 32 人的划船队——同一拍划同一桨，谁也不能自己划自己的。**
有了 warp 这个词，我们就可以打开 SM 看内部了——接下来三页全在 SM 里面。

## 第 7 页｜Inside a Blackwell SM: The Resource Budget（SM 的资源账本）

🎤 刚才说 warp 是硬件的调度单位，那一个 SM 能同时"照看"多少 warp？打开账本，三组数字要记住：
① 每 SM 同时跟踪 **64 个 warp = 2048 个线程**——这是调度器来回切换的"池子"；
② **64K 个 32 位寄存器**（共 256 KB），单个线程最多用 **255 个**；
③ **256 KB 统一的 L1/共享内存**，其中最多 **228 KB** 可配成用户管理的共享内存（实际可用 **227 KB**——CUDA 每 block 保留 1 KB）。
正是这些大额片上预算，让一个 SM 能"玩杂耍"般同时伺候几千个线程而不用频繁下 DRAM。右图（Figure 6-2）就是 SM 的内部结构，下一页细看。
大白话：**SM 是一个车间：寄存器是每人的工具腰带，共享内存是中间的工作台。**

❓ 可能被问："227 和 228 怎么回事？"→ 共享内存可配上限 228 KB，但 CUDA 每 block 保留 1 KB，所以单 block 最多申请 227 KB。

## 第 8 页｜Analogy: One SM Is a Workshop（比喻：SM 车间图）⭐

🎤 上一页的数字有点抽象，我画了一张图。我们把一个 SM 想成一间大车间。车间里有**四个调度分区**，形象地叫 4 个 mini-SM——注意它们**不是**四个独立的 SM，CUDA 程序只能看到整个 SM，不能指定"把这个 block 放到 mini-SM 2"；mini-SM 只是帮助理解内部调度的概念。每个分区有自己的 **Warp Scheduler**，相当于一名调度员。
货架上的**安全帽**：一顶安全帽代表一个完整 warp 的执行状态（不是一个线程）。每个分区约 16 顶 × 4 个分区 = 整个 SM 最多 **64 个常驻 warp** = 2,048 线程。"常驻"的意思是寄存器、状态都已分配好、随叫随到——**不等于 64 个 warp 同一拍都在执行**。
货架下面**被高亮的工人小组**：这一拍被调度员选中（发射指令）的 warp——4 名调度员每拍最多各挑 1 个，所以图中间写"每拍最多选中 4 个 warp"。
左下角单独放大：**1 warp = 32 个线程**，SIMT 下同一条指令、32 个不同的 idx，调度员永远整组整组地派工。
图最底部是共同的地基——Register File、L1、Shared Memory——说明四个分区仍然共用同一个 SM 的片上资源。
大白话：**64 支候选队伍待命，4 名调度员，每拍最多派出 4 支——每支队伍 32 名工人。**

❓ 可能被问："mini-SM 是真实硬件吗？"→ 是真实存在的调度分区（processing block），各有独立的 warp scheduler 和执行单元，但对 CUDA 编程模型不可见、不可指定，所以说它是"理解硬件调度的概念"。
❓ "16 顶安全帽是精确值吗？"→ 64 ÷ 4 的平均示意；实际驻留分布由硬件决定。

## 第 9 页｜Let the Workshop Move（让车间动起来）⭐

🎤 静态的图看懂了，关键是让它**动起来**。假设某个分区此刻：Warp A 在等 HBM 数据、Warp B 在等上一条计算结果、Warp C 和 D 数据都齐了。调度员**不会**傻等 A 和 B——它直接发射就绪的 C 或 D，跳过停顿的 warp 不花任何代价。下一拍它可能又挑别的：周期 1 发 C、周期 2 发 F、周期 3 发 D、周期 4 发 H……
两个精确的说法：①"选中"更准确说是**发射（issue）指令**——指令开始执行，不代表这一拍执行完；② 延迟隐藏**不是**让某个 warp 的等待消失，而是**它等待时让别的 warp 干活**，把等待"填满"。
这就把第 7 页的 64 这个数字讲活了：64 个常驻 warp 的意义，就是给 4 个调度器准备足够深的**就绪候选池**——反过来，占用率低 = 池子浅 = 没得可切 = 等待变成纯空转。这句话记住，第 37 页起的 22 倍案例就是它的实证。
大白话：**调度员从不站着干瞪打盹的队伍——总有另一支队伍的零件已经到货了。**

## 第 10 页｜Two Views: Software vs. Hardware（软硬件两套视角）⭐

🎤 讲到这里把 Part I 的思路捋一遍，这页就是全章的"地图"。我们其实一直在讲**同一台机器的两套词汇**：
**软件视角（你写代码时用的）**：thread / block / grid——这是 CUDA 编程模型，你在 kernel 里只决定 grid 和 block 的大小。
**硬件视角（机器实际运行的）**：SM、warp、调度器——block 会被放到**恰好一个 SM** 上（不会跨 SM 拆开），硬件再把它自动切成 32 线程一组的 warp 交给调度器；grid 则铺满整块 GPU。
对应到工厂比喻：grid = 整座工厂，block = 一间车间的活儿，warp = 32 人小组，thread = 一名工人。**warp 是两套语言的桥**——你写代码时感觉不到它，但性能好坏全在它身上（32 倍数、分化、占用率都是 warp 的事）。
右图顺便预告第三部分的内存工厂：寄存器 = 随身工具箱，Shared Memory = 车间公共工作台，L1 = 车间临时货架，**L2 = 全厂中央中转仓，HBM = 厂外大仓库**；一次全局内存读取依次尝试 L1 → miss 走 L2 → miss 走 HBM，数据按 HBM → L2 → L1 → 寄存器原路返回。
大白话：**软件说的是 grid 和 block；硬件回答的是 SM 和 warp——warp 就是两套词汇表的交汇点。**

❓ 可能被问："block 会不会跨 SM？"→ 不会。一个 block 从生到死都在同一个 SM 上；能跨 SM 协作的是第 14 页要讲的 thread block cluster（DSMEM）。

## 第 11 页｜Four Warp Schedulers, Dual Issue（四调度器、双发射）⭐

🎤 每个 SM 其实是**四个"迷你 SM"**：四个独立 warp scheduler（warp 调度器），各带自己的派发逻辑。两层机制：
① 每个调度器每拍发射一个 warp 的指令 → **每拍最多 4 个 warp 同时推进**；
② **dual-issue（双发射）**：同一拍里同一个 warp 可以同时发一条算术指令（INT32/FP32/Tensor Core）+ 一条访存指令（load/store）。**限制：必须来自同一个 warp，不能跨 warp 拼**。
最好情况：每拍 **4 数学 + 4 访存**指令齐飞。右边表格（Table 6-1）是每拍的上限。注意表格下的小字：书里所有指标表的数值都是**示意值**，真实 benchmark 在配套 GitHub 仓库——这是全书的统一免责声明。
大白话：**四条收银通道，每个收银员一只手扫码（算术）、另一只手装袋（访存）。**

📚 原书精读：
> "You can think of the SM as four 'mini-SMs' sharing on-chip resources." → 把 SM 想成四个共享片上资源的迷你 SM。
> "Note that the dual-issue must come from the same warp—and not across warps." → 双发射必须来自同一 warp。

## 第 12 页｜SFUs and Load/Store Pipelines（特殊功能单元与访存管线）

🎤 SM 里还有两类容易被忽略的部件。左边：**SFU（Special Function Unit，特殊功能单元）**，专算超越函数——sin、cos、倒数、开方。关键点：它有**自己独立的管线**，不占双发射的"数学+访存"名额——慢速复杂运算永远不会堵住核心管线，混合运算的 kernel 因此有更多指令级并行。
右边：**LD/ST（load/store）访存管线**，每 SM 共 16 条（每调度器 4 条），负责读写 L1/共享内存、L2 和全局显存。书里特别警告：**具体管线数量和配对规则不受保证**——判断 kernel 是"访存发射受限"还是"计算发射受限"要靠 profiling，细节查 Blackwell tuning guide。
大白话：**SFU 是商店后面的专柜——复杂业务去那儿办，快速通道保持流动。**

## 第 13 页｜Blocks Cooperate Inside, Stay Independent Outside（块内协作、块间独立）

🎤 两条规则一正一反。**块内**：线程用共享内存交换数据、用 `__syncthreads()` 同步——这是个 barrier（栅栏），所有人到齐才继续。但**每个 barrier 都有开销**，书里明确说：**尽量减少同步点**（右图 Figure 6-6）。
**块间**：完全独立、执行顺序不保证任何东西。这个"不方便"恰恰是 CUDA 可扩展性的来源——调度器可以把 block 随意撒到所有 SM；你的代码在未来 SM 更多的 GPU 上**不改就能跑**。
大白话：**班组碰头会（barrier）有用但贵，能少开就少开；而且永远别假设 A 班组比 B 班组先干完。**

## 第 14 页｜Thread Block Clusters and DSMEM（线程块簇与分布式共享内存）

🎤 传统上不同 block 的线程不能直接协作，现代 GPU 打破了这一点：**thread block cluster（线程块簇）**——一组能**跨 SM 通信**的 block，有簇级硬件 barrier。底层是 **DSMEM（分布式共享内存）**：把参与簇的各 SM 的共享内存 bank 用**片上高速互连**连成一个池子（右图 Figure 6-5）。效果：不同 block 的线程能以**片上速度**读、写、原子更新彼此的共享缓冲——**不花全局显存带宽**。这是今天大矩阵乘、LLM 负载的关键使能技术，第 10 章细讲，今天知道它存在即可。
大白话：**相邻班组在工作台之间的墙上开了个门，零件直接递过去，不用再走仓库。**

📚 原书精读：
> "This unification allows threads in different blocks to read, write, and atomically update one another's shared buffers at on-chip speeds—and without using global memory bandwidth."

## 第 15 页｜Occupancy: The Central Metric（占用率：本章核心指标）⭐

🎤 现在正式定义标题里的概念。**Occupancy = SM 上活跃 warp 数 ÷ 硬件上限（Blackwell 是 64）**。profiler 里实测的平均值叫 **achieved occupancy**。
为什么要高？**latency hiding（延迟隐藏）**——一个 warp 卡在访存上，另一个随时顶上。Blackwell 的大寄存器堆（64K/SM）让高占用率更容易达到。
但立刻要说"制衡"：warp 们**共享寄存器和共享内存**。塞太多 warp → 每线程分到的寄存器变少 → **register spilling（寄存器溢出）**到慢速内存——你亲手制造了新的停顿。所以书里的忠告是：**把 occupancy 和寄存器/共享内存用量放在一起 profile**。
大白话：**occupancy 是车间的工位坐满率——空位浪费"延迟隐藏"，但过度拥挤会导致没人有工具用。**

📚 原书精读：
> "Keeping more warps in flight is known as high occupancy on the SM... when one warp stalls, another is ready to run."

## 第 16 页｜Warp Divergence（warp 分叉）⭐

🎤 SIMT 的"齐步走"有个天生软肋：分支。**同一个 warp 内**如果有人走 `if`、有人走 `else`，硬件只能**串行化**：先蒙住（mask）走 else 的那些"车道（lane）"执行 if 路径，再反过来执行 else 路径（右图 Figure 6-8）。执行时间**乘以分支路径数**。
两个要点：① **跨 warp 无惩罚**——不同 warp 各走各的分支完全免费；② 实用推论：分支条件尽量**按 warp 对齐**——按 `threadIdx.x / 32` 分支无害，按 `threadIdx.x % 2` 分支是最坏情况（每个 warp 都劈成两半）。检测和治理在第 8 章。
大白话：**划船队一半人往左划、一半往右划，船就得把两个动作各做一遍——慢一倍。**

📚 原书精读：
> "warp divergence multiplies the overall execution time by the number of branches."
> "Divergence is an issue only for threads within a single warp."

## 第 17 页｜Hardware Limits I: Warp and Block（硬件上限之一）

🎤 查表页，重点讲右边的框：**为什么必须是 32 的倍数**——一个 **33 线程的 block 要占两个 warp 槽位**，第二个 warp 只有 1/32 的车道干活，却照样占一个完整的调度器名额。每个"不是 32 倍数"的选择都在给虚空捐算力。
表格三行：warp 固定 32；每 block 最多 **1024 线程**（三个维度乘起来 ≤1024）；也就是每 block 最多 32 个 warp。
下面两条补充：block 太大 → 寄存器要得太多 → **溢出**；共享内存也是**SM 上全部常驻 block 共享** 227 KB。反过来，**block 小一点往往 occupancy 更高**——每 SM 能塞更多独立的 block。

## 第 18 页｜Hardware Limits II: SM-Resident and Grid（硬件上限之二）

🎤 每 SM 的常驻（resident）上限：**64 warp、2048 线程、32 block**——64 这个数已经**好几代不变**，所以 occupancy 的经验能跨代延续。
算一笔账（这页最有用的部分）：每 block 1024 线程 → 一个 SM 只能驻 **2 个 block**；改成 256 线程 → 能驻 **8 个**——同样 2048 线程，灵活性完全不同。
grid 上限：X 维约 21 亿个 block，Y/Z 各 65,535；每设备最多 **128 个 kernel 并发**。实践结论：**你永远先撞上 per-SM 限制**；真要超过 Y/Z 上限，用 2D/3D grid 或分多次启动（multilaunch）。

## 第 19 页｜Compatibility: PTX, SASS, and Fatbins（兼容性模型）

🎤 CUDA 生态的护城河之一：**前后向兼容**。三个概念：**SASS**——特定架构的最终机器码（sm_90 = Hopper、sm_100 = Blackwell），只带单架构 SASS 的二进制**上不了新 GPU**；**PTX**——虚拟指令集，驱动在加载时 JIT 即时编译成新架构的 SASS，这就是**前向兼容**的机制；带 f 的家族目标（如 `sm_100f`）只在同特性家族内可移植。
最佳实践（右框）：发 **fatbin（胖二进制）**——通用 PTX + 需要的家族专用 cubin，并为其他架构留 fallback。验证方法：设 `CUDA_FORCE_PTX_JIT=1` 强制走 PTX JIT——**二进制里没有 PTX 的话 kernel 启动直接失败**，逼你重新构建。
大白话：**PTX 是菜谱，SASS 是做好的菜——把菜谱也一起发货，未来的厨房才能重新做。**

---

# Part II：CUDA 编程（第 20–27 页）

## 第 20 页｜Anatomy of a CUDA Kernel: Device Side（kernel 解剖：设备侧）⭐

🎤 全书第一段完整 CUDA 代码的设备侧。四个零件：
① **`__global__`**：跑在 device、从 host 调用；
② 三个内置变量拼出**全局唯一编号**：`idx = blockIdx.x * blockDim.x + threadIdx.x`——blockIdx 是"我在第几个班组"、blockDim 是"班组多大"、threadIdx 是"我是几号"；
③ **`if (idx < N)` 边界检查**（第 23 页专门讲为什么）；
④ host 侧的启动语法 **`<<<blocksPerGrid, threadsPerBlock>>>`**——任何 kernel 调用的两个核心参数。
最重要的思维转变：**你写的是"一个工人的作业说明书"，CUDA 复印一百万份，每份发一个不同的行号。**

## 第 21 页｜Host Side: The Six-Step Data Flow（主机侧：六步数据流）

🎤 host 侧完整代码，六步在注释里标了号：**① 分配**——注意 `cudaMallocHost` 分配的是 **pinned（页锁定）内存**，不会被操作系统换页，这是后面异步拷贝能真正重叠的前提；**② H2D 拷贝**；**③ 启动**——256 线程/block，`(N+255)/256` 向上取整（N=100 万时 = 3907 个 block）；**④ `cudaDeviceSynchronize()` 等设备完成**；**⑤ D2H 拷回**；**⑥ 清理**。
两个习惯请照抄：`h_` 前缀 = host 指针、`d_` 前缀 = device 指针——全书通用。书里注明这段**还没有优化**——它是后面全书持续改进的"简单、完整的模板"。

## 第 22 页｜Why Pass N?（为什么要传 N）

🎤 初学者常见疑问：kernel 为什么不能自己看数组多长？蓝框是书里的原文回答：**CUDA kernel 的设计就是"在单个线程内工作、与几千个线程并肩、处理输入数据的一个分区"——N 定义了分区的边界。** CPU 函数可以问容器要 size；kernel 拿到的是**裸指针**，必须被告知世界的尽头在哪。配合三个内置变量，N 让每个元素被**干净且唯一**地并行处理。
大白话：**一百万份复印的作业说明书内容相同，N 是上面写"活到哪儿为止"的那一行。**

📚 原书精读：
> "a CUDA kernel function is designed to work inside of a single thread, alongside thousands of other threads, on a partition of the input data."

## 第 23 页｜The Bounds Check — and Lazily Surfacing Errors（边界检查与懒惰报错）⭐

🎤 左边是书里 N=63 的推演：63 个元素，调度器派 **2 个 warp**（64 线程）。第一个 warp 处理 0–31 没问题；第二个 warp 的最后一个线程如果不检查就会**读越界地址** → `cudaErrorIllegalAddress`。书里的金句：CUDA kernel 里到处是边界检查，**"如果没看到，你应该弄明白它为什么不在。"**
右边讲 CUDA 报错的反直觉机制：kernel **异步执行、没有每线程异常**——非法访问只是给整次 launch 设一个**全局故障标志**，host **要等下一次同步或 CUDA API 调用**才看到——错误是**懒惰浮现**的。规范写法：启动后紧跟 `cudaGetLastError()` + `cudaDeviceSynchronize()`。
大白话：**GPU 出事不会给你打电话——它留张字条，你下次开信箱才看到。**

## 第 24 页｜Choosing Launch Parameters: The 256 Recipe（256 食谱）

🎤 为什么从 256 开始？书里给了四个理由：① **32 的倍数**——没有半空 warp 占调度器名额；② **延迟隐藏**——8 个 256 线程的 block 正好填满 SM 的 2048 容量，不过度订阅；③ **occupancy**——8 warp/block 通常不会把寄存器和共享内存用爆；④ **资源均衡**——离 1024 上限远、调整余地大。Blackwell 的建议区间：**256–512**。grid 公式 `(N+255)/256` 向上取整保证全覆盖。
大白话：**256 是 block 尺寸里的"中杯咖啡"——几乎不会点错，之后按 profiling 微调。**

## 第 25 页｜2D and 3D Kernel Inputs（二维与三维输入）

🎤 图像这类天然二维的数据，用 2D grid × 2D block。左边代码三处变化：坐标变成 `x` 和 `y` 两个（用 `.x` 和 `.y` 分量）；边界检查变成 `if (x < width && y < height)`；再用 `idx = y * width + x` 摊平成一维下标访问。host 侧用 `dim3` 类型——比如 1024×1024 的图配 16×16 的 block（还是 256 线程）。同一套路用 `dim3(x,y,z)` 直接推广到 3D 体数据。书里说明：全书大多用 1D 或 2D（tiled 分块）配置，1D 时用普通 int 就行。
大白话：**同一份食谱、两个坐标轴：每个工人的工牌从"行号"换成"（行，列）"。**

## 第 26 页｜Allocate Asynchronously（异步分配：流与内存池）⭐

🎤 隐藏成本警告：`cudaMalloc`/`cudaFree` 是**同步且贵**的——全设备同步 + 操作系统调用（mmap/ioctl）+ 内核态切换。训练循环里每轮分配释放，积少成多。
解法三步（左边代码）：建**非阻塞 stream**（stream = GPU 上的"传送带"，同带内按序、异带互不干扰）→ `cudaMallocAsync` 在带上分配 → `cudaFreeAsync` 释放。底层是**内存池**：释放的内存回池等复用，不找 OS 要新的——省系统调用、减少**碎片化**。关键：`cudaFreeAsync` **只等自己这条 stream**，没有全局同步。
书里还有个提示：用 `cudaStreamNonBlocking` 建流是为了避开**老式默认流的隐式全局屏障**（第 11 章展开多流重叠）。
大白话：**在停车场养一支车队、用时拿钥匙——别每次送货都买车再卖车。**

## 第 27 页｜Tuning the Pool; PyTorch's Caching Allocator（池调优与 PyTorch 分配器）

🎤 两个进阶旋钮：**`cudaMemPoolAttrReleaseThreshold`**——池子保留多少内存不还给系统；**`cudaMemPoolTrimTo`**——主动归还。权衡的是"总显存占用"和"碎片化"。
右边是跟大家日常最相关的连接：**PyTorch 的 caching allocator**（配置项 `PYTORCH_ALLOC_CONF`，旧名 PYTORCH_CUDA_ALLOC_CONF）就是同一思路——复用显存、避免每建一个 tensor 都调一次同步的 cudaMalloc。
书里的选型结论：一次性缓冲区用阻塞版没问题；**分配密集的长循环用 async + 池**，性能更稳、吞吐更高。

---

# Part III：内存层级（第 28–36 页）

## 第 28 页｜The Memory Ladder at a Glance（内存梯子总览）⭐建议讲 2 分钟

🎤 全章核心表格（Table 6-5），从上往下**容量越来越大、速度越来越慢**：寄存器（单周期、几十 TB/s）→ 共享+L1（20–30 拍、TB/s）→ TMEM（Tensor Core 专用）→ 常量缓存（1 拍广播）→ L2（**126 MB**、约 200 拍）→ local memory（寄存器溢出区，实际在 DRAM！）→ HBM3e（**180 GB、约 8 TB/s**、几百到一千拍）。
一句话行动准则：**能复用就往上层放，必须下 HBM 就合并访存。**接下来 8 页逐层拆开讲。
大白话：**书桌（寄存器）→ 办公室书架（共享内存）→ 楼里图书馆（L2）→ 城另一头的仓库（HBM）。尽量在书桌上干活。**

📚 原书精读：
> "maximizing data reuse in registers, shared memory, and L1/L2 cache—and minimizing reliance on global memory—is essential for high-throughput GPU kernels."

## 第 29 页｜Registers — and the Spill Cliff（寄存器与溢出悬崖）

🎤 书里的说法很形象：每个线程"从寄存器堆开始它的旅程"——单周期读写、几乎不与任何东西争抢、每 SM 几十 TB/s。预算：64K/SM、**每线程最多 255 个**。
然后是**悬崖**：需要更多（局部变量太多、编译器临时量太多）→ 溢出（spill）进 **local memory**——名字里有 local，物理上在**片外 DRAM**，几百到一千多拍。这是 CUDA 里最经典的"静默性能杀手"。监控指标：Nsight Compute 的 **Registers Per Thread**。
大白话：**"local memory" 是 CUDA 里最误导人的名字——它是城另一头的自助仓库，不是你的口袋。**

## 第 30 页｜Shared Memory + L1: One SRAM, Two Jobs（共享内存与 L1）

🎤 一块 **256 KB 的 SRAM 干两份工**：用户管理的共享内存（上限 228/227 KB）+ L1 数据缓存。分割比例（carveout）你自己选——左边代码 `cudaFuncSetAttribute(...PreferredSharedMemoryCarveout...)`。
性能：20–30 拍；避开 **bank conflict（存储体冲突）**——多个线程撞到同一个 bank 会串行化——就能拿到 TB/s 级吞吐。这里是**块内协作的工作台**：矩阵分块（tile）、归约、数据暂存都在这儿。
大白话：**一张工作台，隔板可调：多少归你们班组的项目台，多少归自动的缓存架。**

## 第 31 页｜TMEM and TMA: Feeding the Tensor Cores

🎤 Blackwell 新增：**TMEM**，每 SM 256 KB 专用 SRAM，是第五代 Tensor Core 指令（tcgen05、**UMMA**——统一矩阵乘累加）的**累加器**，与 Tensor Core 之间几十 TB/s。
特别之处：**CUDA C++ 里拿不到它的指针**——数据进出全由 **TMA（张量内存加速器）**按"描述符"自动编排。看右图的 C = A×B：操作数 B 在共享内存、A 和累加器在 TMEM；TMA 负责 HBM→L2→SMEM 的搬运，SMEM↔TMEM 由 Tensor Core 指令隐式完成。净效果：**大幅减少 Tensor Core 对全局显存的依赖**。细节第 10 章。
大白话：**TMA 是专职叉车队，在后台不停搬运物料，做矩阵乘的大厨永远不用离开厨房。**

## 第 32 页｜Constant Cache: One-Cycle Broadcast（常量缓存：单拍广播）

🎤 每 SM 约 8 KB 的小缓存，前置 64 KB 的 `__constant__` 只读空间。绝活：**全 warp 32 个线程读同一个地址时，1 拍广播给所有人——和读寄存器一样快**。反之，分歧读会跨车道串行化。所以它适合**小、只读、所有线程访问同一地址**的数据。
右边绿框是书里点名的 LLM 场景（都是高频小表）：**RoPE 旋转位置编码查找表、ALiBi 斜率、LayerNorm 的 γ/β 向量、embedding 量化 scale**——全体线程共享、零全局显存流量。
大白话：**它是喇叭不是信箱：一次广播全班 32 人都听到——前提是大家问的是同一个问题。**

## 第 33 页｜L2 Cache: The GPU-Wide Middleman（L2：全 GPU 的中间人）

🎤 **126 MB**、所有 SM 共享，是片上 SRAM 和片外 HBM 之间的胶水。约 200 拍、聚合带宽几十 TB/s，吸收 L1 溢出。最有价值的性质：**跨 block 复用**——一个 block 取过的数据，其他 block 从 L2 拿，不用重访 DRAM。
右框是书里的合并访存法则：**把全局加载组织成 128 字节对齐的 coalesced（合并）段**，干净映射到缓存线——避免事务被拆分，同时拉满 L2 和 DRAM 带宽。具体手法第 7 章。
大白话：**L2 是楼里的图书馆：同事已经从仓库借来的书，你从馆里拿就行——前提是大家都按"整架"借书，不要一页一页借。**

## 第 34 页｜Global HBM3e, the Dual-Die B200, and Coherency

🎤 梯子最底层：**HBM3e**，B200 是 180 GB（B300 约 288 GB）、约 8 TB/s——容量最大、带宽惊人，但**延迟几百到一千多拍，是链条上最慢的一环**。寄存器溢出、超大自动数组（local memory）也在这里付同样的价。
两个冷知识：① 书里的边栏——**B200 物理上是两块 die**（受光刻极限限制），10 TB/s 片间互连、各接 4 个 HBM 栈，但呈现为**一个 GPU、一个地址空间**；② **point of coherency（一致性生效点，Figure 6-15）**：内存一致性按 thread → block → cluster → device → system 五级建立——**通信范围越广，代价越高**。

## 第 35 页｜Unified Memory: Convenience with a Catch（统一内存：便利有价）

🎤 `cudaMallocManaged()`：CPU+GPU 一个一致的地址空间——不用分开管缓冲、不用手写 memcpy。底层机制：页**按需迁移**。
"catch"（代价）在这里：GPU 摸到一个还在 CPU 内存的页 → **缺页（page fault）、kernel 停等**。硬件差异巨大：PCIe 上按缺页搬运**可能比手动 memcpy 还慢**；Grace 超级芯片的 **NVLink-C2C 约 900 GB/s**，迁移接近原生速度——**但延迟永远不是零**。
大白话：**统一内存是客房服务——方便，但不提前点单，开饭时就得饿着等。**

📚 原书精读：
> "any unexpected page-fault during a kernel launch will stall the GPU while the runtime moves the needed page into place."

## 第 36 页｜Taming Unified Memory: Prefetch, Advise, Attach（驯服统一内存）

🎤 治理"意外缺页"的三板斧（左边代码从上到下）：
① **`cudaMemPrefetchAsync`**——kernel 启动**前**整体预取，把"首次触碰迁移"变成可重叠的异步传输；
② **`cudaMemAdvise` 三条建议**：`SetPreferredLocation`（数据主要在哪用）、`SetReadMostly`（基本只读，驱动可以两边各放副本）、`SetAccessedBy`（让另一块 GPU 直接映射、不触发迁移）；
③ **`cudaStreamAttachMemAsync`**——把一段内存绑定到一条 stream，别的 stream 不再因它意外停顿。
补充：没有 NVLink-C2C 的多卡系统，用 peer copy/预取把数据钉在 NUMA 本地。书里的结论：三板斧用齐，统一内存性能**非常接近手动 cudaMemcpy**，同时保住简洁性。

---

# Part IV：Occupancy 实战（第 37–45 页）

## 第 37 页｜Occupancy Ground Rules（占用率基本法则）⭐

🎤 蓝框是书里"CUDA 性能最基本的法则"原文：**"Launch enough parallel work to fully occupy the GPU."**——启动足够多的并行工作填满 GPU。
下面两条规则务必分清：**规则一**——occupancy 低且性能差：第一味药是**加并行度**，把 occupancy 推向 **80–100%**；**规则二**——occupancy 已经中高但 kernel 是 memory-bound：推到 100% **没用**——你只需要"刚好够藏延迟"的 warp 数，之后瓶颈在带宽。
接下来 5 页是一个完整案例：同一个操作（C = A + B，一百万元素）、两种实现、profiler 的裁决。
大白话：**规则一是把座位坐满；规则二提醒你：坐满的公交堵在路上还是堵着。**

## 第 38 页｜Case Study, Take 1: addSequential（案例上：串行版）

🎤 反面教材：只有 `blockIdx.x==0 && threadIdx.x==0` 的**一个线程**for 循环加完一百万个元素，启动配置 `<<<1,1>>>`（书里注明：这个 kernel 的语义就是单线程，别改启动配置）。
后果（书中原话）："GPU 的庞大资源基本闲置……只有一个 warp、甚至只有其中一个线程在干活。"更致命的是：**没有其他 warp 可切换 → 每次访存等待都是纯粹的空转 → 延迟隐藏为零**。
大白话：**你租下整座工厂，雇了一个工人。**

## 第 39 页｜The Same Trap in PyTorch（PyTorch 里的同款陷阱）⭐

🎤 同样的错误在 PyTorch 里更常见也更隐蔽——左边代码：用 Python for 循环逐元素 `C[i] = A[i] + B[i]`。这会**串行发射一百万个迷你 kernel**，书里的说法是把 GPU 用成了"标量、非并行处理器"。occupancy 跟 addSequential 一样惨，还外加**一百万次 kernel 启动的 CPU 开销**。
书里的忠告：除非你在写全新的东西，**几乎总有现成的 PyTorch 原生向量化实现**——包括 PyTorch 编译器生成的代码。**不要在 GPU 操作外面套 Python 循环。**
大白话：**GPU 上最常见的性能 bug，是不小心把 GPU 当成了一台很贵的单核 CPU。**

## 第 40 页｜Case Study, Take 2: addParallel — and C = A + B（案例下：并行版）

🎤 正确姿势：每线程加**一个**元素，`<<<(N+255)/256, 256>>>` 启动约 3907 个 block、一百万个线程（Figure 6-19）。代码里两个细节：**`__restrict__`** 注解承诺指针无别名（aliasing），解放编译器优化；书中完整版还用了 Part II 教的全套习惯——pinned 内存、非阻塞 stream、cudaMallocAsync/cudaMemcpyAsync 全异步链路。
PyTorch 版就一行：**`C = A + B`**——单个向量化 kernel，海量线程并行。
大白话：**同一座工厂，现在每个工位都有人——PyTorch 版更省事：直接找劳务派遣公司。**

## 第 41 页｜Measuring It: nsys and ncu（量化验证的命令）

🎤 书里给了完整命令行（左边），分工记一句话：**nsys 回答"时间去哪了——GPU 是被饿着还是被堵着"；ncu 回答"这个 kernel 为什么慢——occupancy？停顿？缓存？"**。必须**两个都跑**：只跑 nsys 看不到 kernel 内部低效；只跑 ncu 不知道 kernel 是否被及时喂数据。
一个彩蛋：ncu 命令里那个很长的指标名 `sm__warps_active.avg.pct_of_peak_sustained_active` **就是 achieved occupancy 本尊**——面试可以拿出来说。
大白话：**nsys 是整台机器的秒表，ncu 是对准一个 kernel 的显微镜。**

## 第 42 页｜Results: 22× from Occupancy Alone（结果：22 倍）⭐

🎤 裁决（Table 6-6，示意值）：**kernel 时间 48.21 → 2.17 ms，22 倍**；GPU 利用率 1.5% → 95%；achieved occupancy 1.3% → **38.7%**；warp 执行效率 3.1% → 100%——3.1% 就是 1/32：一个 warp 里只有一条车道干活。
两个洞察：① **38.7% 的占用率就买到了 22 倍**——根本不需要 100%；② 右图（Figure 6-20）是加速的本质：串行版时间线是"操作+访存空档"交替；并行版**空档被其他 warp 的工作填满**——延迟隐藏的可视化。
补充：不同工具叫法不同——nsys 的 "GPU Utilization" 和 ncu 的 "SM Active %" 反映的是同一件事。

📚 原书精读：
> "No matter how fast each thread is, you need lots of threads to leverage the GPU's throughput potential."

## 第 43 页｜Reality Check: Memory-Bound Workloads（现实检验：LLM decode）⭐

🎤 泼冷水的一页，也是通往 roofline 的桥。占用率之后的下一层是每 warp 的效率（ILP，第 8 章）——**但即使 100% occupancy，memory-bound（受限于数据搬运）的 kernel 照样受损**。
书里的典型例子：**LLM 的 decode 阶段**——每生成一个 token 都要把**模型权重**从 HBM 流进寄存器/共享内存。几千亿参数 × 约 1 字节 ≈ **几百 GB 一遍**——不管开多少线程，显存带宽先饱和。
下面的框是书里的趋势判断：**GPU FLOPS 的增速正在甩开显存带宽**（HBM3e 约 8 TB/s，但算力和模型规模长得更快）——优化数据搬运在现代 AI 负载里绝对关键。
大白话：**当整个活儿就是"从仓库搬箱子"，在办公桌前多雇职员没有用——这正是 Part V 的 roofline 模型要形式化的事。**

## 第 44 页｜__launch_bounds__: Compile-Time Occupancy Control

🎤 第一件调优工具，编译期的。`__launch_bounds__(256, 16)` 给编译器两个信息：**承诺** block 不超 256 线程；**请求**每 SM 至少驻 16 个这样的 block。编译器于是**压缩每线程寄存器、克制展开和内联**，好塞进更多 warp。
有意思的细节：16×256 = 4096 **超过了** 2048 的硬件上限 → 编译器**削到 8 个 block** 并给出 ptxas 警告（".minnctapersm will be ignored"）——说明这些参数是愿望，硬件上限说了算。
交易的本质：**牺牲一点单线程性能，换更多 warp 在飞、更稳定的 warp 吞吐**。危险区：压得太狠 → 寄存器不够 → **溢出到 local memory**——比不优化还慢。
大白话：**拿"每个工人的工具箱大小"换"车间里能站多少工人"——甜点只能靠实验找。**

## 第 45 页｜The Occupancy API: Runtime Autotuning

🎤 第二件工具，运行时的。`cudaOccupancyMaxPotentialBlockSize()` 根据 kernel **实际的**寄存器/共享内存消耗，自动算出 occupancy 最优的 block size。
书里点名的两个坑：① 返回的 **`minGridSize` 是"喂饱 occupancy 的最小 grid"，不是覆盖 N 个数据的 grid**——真正的 grid 要取 `max(minGridSize, ceil(N/blockSize))`；② kernel 用了 `extern __shared__` 动态共享内存的话，**字节数必须如实传**。
最后是书里的边栏忠告：**API 的建议要用 ±1–2 档 block size 实测验证**——寄存器压力和 L2 行为可能让"略低于最大占用率"的配置实际更快。

---

# Part V：正确性与 Roofline（第 46–49 页）

## 第 46 页｜Compute Sanitizer（计算消毒器）

🎤 换个话题：不谈快慢，谈对错。几万个线程的程序，传统 debugger 抓不住偶发的内存错误和竞争。CUDA Toolkit 自带的 **Compute Sanitizer** 运行时插桩，四件套：
左栏（内存类）：**memcheck**——越界/未对齐/泄漏（最常用）；**initcheck**——读了未初始化的显存（典型病因：**忘了 H2D 拷贝**）。
右栏（并发类）：**racecheck**——共享内存竞争（WAW/WAR/RAW 三种冒险）；**synccheck**——非法同步、错配 barrier → 死锁。
用法：`compute-sanitizer --tool <名> ./app`，`--kernel-name` 过滤、NVTX 标注。书里的最佳实践：**进 CI + `--error-exitcode`**——正确性回归在合码前拦下。
大白话：**十万个线程面前，"我这儿跑过没问题"毫无意义——sanitizer 是你的安全带。**

## 第 47 页｜The Roofline Model（屋顶线模型）⭐建议讲 2–3 分钟

🎤 最后一个大概念，也是全书反复用的分析框架（右图 Figure 6-21）。
横轴：**arithmetic intensity（算术强度）**= 每从 HBM 搬 1 字节做几次浮点运算（FLOPs/byte）。两条天花板：**水平的计算屋顶**（约 80 TFLOP/s FP32）和**倾斜的内存屋顶**（约 8 TB/s），交点是 **ridge point（脊点）≈ 10 FLOPs/byte**（80T ÷ 8T）。脊点**左边 = memory-bound**（饿数据），**右边 = compute-bound**（饿计算）。
现场算一遍书里的例子：C = A + B——读 8 字节、加 1 次、写 4 字节 → **1 FLOP ÷ 12 字节 ≈ 0.083**——离脊点差 100 多倍，**无可救药地 memory-bound**。这就从数学上解释了第 43 页：为什么占用率救不了它。
对到日常：书里的边栏说 LLM 两个阶段各占一边——**prefill 偏 compute-bound、decode 偏 memory-bound**（第 15–18 章展开）。
大白话：**roofline 在你动手前先问一个问题：这个 kernel 是饿计算，还是饿数据？答错方向，白干。**

## 第 48 页｜Moving Right on the Roofline: Lower Precision Pays Twice（低精度一石二鸟）

🎤 知道自己 memory-bound 了怎么办？**往右移**：每字节多干活（片上复用、算子融合），或者最直接的——**把字节变小**。
关键账目：GPU 访存以 **128 字节一个事务**为单位，能装 **32 个 FP32 = 64 个 FP16 = 128 个 FP8 = 256 个 FP4**。FP32→FP16，算术强度**立刻翻倍**；FP8 相对 FP16 再翻一倍吞吐、再省一半内存。Blackwell 原生支持 FP8/FP4 Tensor Core。
更妙的是**硬件解压**：权重以压缩形式存 HBM（甚至 4 位/2 位方案），硬件读取时**在线解压**、再转 FP16/FP32 做高精度累加——**变相扩大可用带宽**。这就是 Blackwell 跑 memory-bound 的 token 生成特别强的架构原因。
蓝框收束：**精度既是计算优化，更是带宽优化——字节减半 ⇒ 强度翻倍 ⇒ 靠近计算屋顶。**

## 第 49 页｜Profiling Workflow: Diagnose, Fix, Re-measure（诊断-修复-复测）

🎤 方法论收尾。**memory-bound 的指标签名**：ncu 里 **DRAM 利用率高 + ALU 利用率低**（warp 停在访存上），**global load efficiency** 下降说明 DRAM 请求满足得不够快；nsys 时间线上 kernel 之间出现空闲段 = GPU 在等数据。
**重叠失败的两大病因**（右图 Figure 6-22 是期望的重叠效果）：① 不想要的**默认流同步**；② **缺 pinned 内存**——没有它 `cudaMemcpyAsync` **根本无法**和 kernel 重叠，这是书里点名的常见性能问题。
修好之后的样子：空闲段消失、memory pipe utilization 爬向峰值、端到端吞吐跳升。最后一条原则贯穿全书：**每改一处，测一次。**

---

# 收尾（第 50–52 页）

## 第 50 页｜Key Takeaways（书中八条要点）

🎤 （左右两栏各念四条，每条一句话）左栏：SIMT——32 线程锁步，多 warp 在飞才藏得住延迟；层级——thread→block→grid，少用 barrier；occupancy 与上限——32 的倍数、记住每 SM 四个数（64 warp/32 block/228 KB/255 寄存器）；启动参数——256 起步、ceil 公式、按 profiling 调。
右栏：异步内存——Async API+stream+池，PyTorch 分配器同理；内存梯子——上层复用、下层合并；统一内存——prefetch+advise 消灭意外停顿；roofline——FLOPs/byte 定战场，低精度+硬件解压往右移，**TMEM+UMMA 能把 kernel 从 memory-bound 拉向 compute-bound**。

## 第 51 页｜Conclusion and What's Next

🎤 整章一句话（蓝框）：**让 GPU 忙起来（occupancy）、让数据离计算近（内存层级）、让 roofline 告诉你下一仗往哪打。**
书的结论里有个反直觉的提醒，必须带到：**占用率最大化不总是最优**——每线程有足够 ILP（指令级并行）时，中低占用率也能跑满吞吐；有时**故意少开线程、让每线程多拿寄存器**反而更快。唯一的裁判是 benchmark。
预告：第 7–8 章讲访存模式和 warp 效率、第 9 章算术强度、第 10 章 TMA 和 warp specialization。我负责的另一章——**第 12 章**——与今天正好衔接：今天解决"单个 kernel 怎么快"，那章解决"kernel 之间怎么编排、让 GPU 永远不用等 CPU"。谢谢大家，欢迎提问。

📚 原书精读（结论段关键句）：
> "However, maximizing occupancy does not guarantee best performance in every case. GPUs can often achieve very high throughput at moderate or even low occupancy if threads have sufficient instruction-level parallelism (ILP)."

## 第 52 页｜References & Further Reading

🎤 参考资料：原书第六章；NVIDIA 的 Blackwell tuning guide 和 CUDA C++ Programming Guide（所有硬件上限的权威出处）；roofline 原始论文（Williams/Waterman/Patterson，CACM 2009）；Compute Sanitizer 和 Nsight 官方文档。表格里的数字是书中示意值，真实分架构 benchmark 在书的 GitHub 仓库。

---

# 附录 A｜答辩预备（高频问题速查）

1. **"线程按什么规则分进 warp？"** 按线程编号连续切：threadIdx 0–31 第一个 warp、32–63 第二个。所以按 `threadIdx.x/32` 分支无害、按 `%2` 分支最坏。
2. **"occupancy 是不是越高越好？"** 不是。够藏延迟即可；memory-bound 时无效；ILP 充足时低占用也能满吞吐；有时少线程多寄存器更快。实测为准（书中结论原话见第 51 页）。
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
