# 第六章 Part III 讲稿（预览稿，对应 chapter06-part3-draft.pdf，共 10 页）

> **说明**：Part III = 原书 Understanding GPU Memory Hierarchy（p240）+ Unified Memory（p246）两个小节：内存阶梯总览 → 寄存器 → 共享内存/L1 → TMEM/TMA → 常量缓存 → L2 → HBM 与双芯粒 → 统一内存及其驯服。页码为预览稿编号。
> 每节结构：🎤 口播稿（覆盖页面全部要点）／📖 页面对照（幻灯片文字的中文版）／大白话／❓ 预备问答。
> **全程复用 Part I 第 7 页的工厂图词汇**：工具箱（寄存器）→ 车间货架（L1）→ 公共工作台（Shared）→ 全厂中转仓（L2）→ 厂外大仓库（HBM）。

---

## 第 1 页｜标题页

🎤 进入 Part III。转场词（对应第 2 页顶部灰字）："我们已经知道怎样产生足够多的线程。但这些线程大部分时间究竟在等什么？答案通常是**数据**——所以接下来看 memory hierarchy。"提醒大家回忆 Part I 那张工厂图：这一部分就是把图的**右半边**（货架、工作台、中转仓、大仓库）逐层打开。

---

## 第 2 页｜The Memory Ladder at a Glance（内存阶梯总览）⭐建议讲 2 分钟

🎤 先看全景表（表 6-5），一行一行往下，规律只有一条：**越往下容量越大、速度越慢**——每一级都在拿容量换延迟和带宽。带大家扫一遍：
**寄存器**——线程私有，每 SM 共 256 KB，约 1 拍，几十 TB/s：最快的一层，Part I 讲过"从大柜子里划格子"。
**共享内存 + L1**——SM 级，可用 228 KB，20–30 拍，TB 级带宽。
**TMEM**——SM 级，256 KB，Tensor Core 专用，约 10 拍。
**常量缓存**——64 KB 空间配 8 KB 缓存，命中且全 warp 同址时**一拍广播**。
**L2**——全 GPU 共享，**126 MB**，约 200 拍，数 TB/s。
**本地内存（溢出区）**——名字骗人：线程私有没错，但**落在 DRAM 上**，≤1000 拍——寄存器装不下的东西掉到这里。
**全局 HBM3e**——B200 上 **180 GB、约 8 TB/s**，但延迟也是几百到 1000+ 拍。
右图（图 6-10）**不是表的重复，分工不同**：左表是"参数表"——每层**多大多快**；右图是"平面图"——每层**在哪、怎么连**，而且画了表里没有的三样东西：另一块 GPU（NVLink 相连）、PCIe、底部的 **CPU host memory**。讲图时用手指三个细节：① SMEM 和 L1 画在**同一个框里、中间一条虚线**——第 4 页"一块 SRAM 两份工作"的视觉预告，隔板就是那条虚线；② **TMEM 单独一个框**——第 5 页的主角先混脸熟；③ **PCIe 连向 CPU host memory 的那条线**——第 9–10 页统一内存的页迁移走的就是它，NVLink 则是多 GPU 章节的伏笔。
一句话收拢："左表告诉你每层楼多大多快，右图告诉你楼盖在哪、楼间的路怎么修。"这页的任务只是建立地图，后面 8 页逐层展开。

📖 页面对照（表 6-5 各列）：Level 层级 / Scope 作用域 / Capacity 容量 / Latency 延迟 / BW 带宽；"Every level trades capacity for latency/bandwidth." → 每一级都在用容量换延迟/带宽。

大白话：**随身工具箱 → 公共工作台 → 全厂中转仓 → 远处大仓库。活儿尽量在自己工位上干。**

❓ 可能被问："DRAM/SRAM 到底是什么？"→ 先拆词：**D**ynamic **R**andom **A**ccess **M**emory，动态随机存取存储器——**Random Access（随机存取）**指可以直接跳到任意地址读写（不像磁带要从头卷到尾）；**Dynamic（动态）**指每个 bit 存在微型电容里、会漏电、必须每秒刷新几千次。两种存储技术对照：**SRAM**（静态）用 6 晶体管触发器存位，极快但占面积、贵——**片上的一切**（寄存器、shared/L1、TMEM、L2）都是它；**DRAM**（动态）用电容存电荷，会漏电所以要不停刷新（"动态"由此得名），密度高、便宜——**片外的一切**（HBM3e、CPU 内存条、local memory 的落点）都是它。HBM 就是把 DRAM die 立体堆叠贴在 GPU 旁边。图 6-10 里 L2 下面那个 "DRAM" 框 = 本卡的 HBM。整个阶梯的分界线就一条：片上 SRAM / 片外 DRAM。
❓ 可能被问："为什么不把所有内存都做成寄存器那么快？"→ 物理规律：越快的存储（SRAM 触发器）单位容量越占面积、越耗电。8 TB/s 的 HBM 已经是堆了十几层 DRAM die 的结果。阶梯不是设计缺陷，是物理约束下的最优解——软件的任务就是把热数据留在上层。

---

## 第 3 页｜Register Pressure Can Turn Fast Storage into DRAM Access（寄存器与溢出悬崖）⭐

🎤 标题就是结论：**寄存器压力能把最快的存储变成 DRAM 访问**。四个要点：
① 每个线程的数据之旅都从**寄存器堆**出发：单拍读写、几乎不与任何人抢，每 SM 几十 TB/s——回忆 Part I：工具箱是从车间大柜子里划走的一格。
② 预算：每 SM 64K 个 32 位寄存器，**单线程最多 255 个**。
③ **悬崖在这**：局部变量太多、编译器临时量太多，超出的部分**溢出（spill）到 local memory**——名字里带"本地"，实际住在**片外 DRAM**：从 1 拍直接掉到几百上千拍，这就是"悬崖"两个字的意思。右图（图 6-12）画的就是这个溢出区。
🗣 **词义先讲一下（spill）**：spill 本义是"（液体）洒出来"——杯子满了，多的水洒到桌上。这里的"杯子"是每线程最多 255 个寄存器；装不下的变量被编译器"洒"到 local memory。工厂话术：**工具箱装满了，多出来的工具寄存到城另一头的自助仓库，每次用都得跑一趟**。一句话记住它和 DRAM 的关系：**spill = 数据从片上 SRAM（工具箱）被挤到片外 DRAM（城外仓库）——这就是它是性能杀手的原因**。
④ 怎么防：盯住 Nsight Compute 里的 **"Registers Per Thread"** 指标——**溢出是无声的杀手**，代码不报错、结果全对，只是莫名其妙地慢。
和 Part I 的呼应：`__launch_bounds__`（Part IV 会讲）就是主动限制每线程寄存器数来换占用率的工具——那是硬币的另一面。

📖 页面对照：
- "Every thread 'begins its journey' at the register file: single-cycle reads/writes" → 每个线程的数据之旅从寄存器堆开始：单拍读写。
- "overflow spills into local memory --- which despite the name lives in off-chip DRAM" → 溢出进 local memory——名字唬人，实际在片外 DRAM。
- "spills are silent killers" → 溢出是无声的杀手。

大白话：**"local memory" 是 CUDA 里最误导人的名字——它是城另一头的自助仓储，不是你的口袋。**

❓ 可能被问："怎么知道自己 spill 了？"→ 两个办法：编译时 `nvcc -Xptxas -v` 会打印每 kernel 的寄存器数和 spill stores/loads 字节数；运行时 ncu 看 Registers Per Thread 和 local memory 流量。看到 spill 字节数 > 0 就要警惕。

---

## 第 4 页｜Shared Memory + L1: One SRAM, Two Jobs（共享内存与 L1：一块 SRAM 两份工作）⭐

🎤 这页的大意一句话：**每个 SM 有一块速度很快的片上 SRAM（256 KB），它既可以当硬件自动管理的 L1 Cache，也可以当程序员管理的 Shared Memory**（最多 228 KB）。Shared Memory 让**同一个 block 的线程共享和复用数据**，20–30 拍、TB 级带宽。
重点讲蓝框：**Shared Memory 为什么能加速——装一次、用很多次**。以矩阵乘法为例：一个 block 负责算一个**输出 tile**。不用 Shared Memory 时，很多线程会**反复从 Global Memory 读同一段数据**——线程 0 读 A 的一段，线程 1 又读同一段，线程 2 又读一遍……全是几百拍的重复搬运。用上之后，数据流变成：**Global Memory →（每份数据只装载一次）→ Shared Memory →（block 内所有线程反复复用）→ 计算**。同一段 A 从"人人下仓库"变成"一人取回、全组共用"，全局内存流量直接除以复用次数。
两个代价（最后一条 bullet）：① **用得太多 → SM 上能同时驻留的 block 变少**——共享内存是按 block 分摊的配额，占用率会掉（Part I 的账）；② **访问布局不好 → bank conflict**，访问被串行化。
工厂话术照旧：这就是**公共工作台**——零件从大仓库领一次放到台上，全组围着用。

📖 页面对照：
- "One 256 KB SRAM per SM, two jobs: hardware-managed L1 cache + programmer-managed shared memory" → 一块 SRAM 两份工作：自动的 L1 + 手动的共享内存。
- 蓝框 "Why it speeds things up: load once, reuse many --- Global → loaded once → Shared → reused by every thread in the block" → 加速原理：装一次、全 block 复用。
- "Two costs: use too much ⇒ fewer resident blocks (occupancy); bad access layout ⇒ bank conflicts" → 两个代价：占用率下降、bank 冲突。

大白话：**一张工作台，隔板可调：多少归你们班组的项目台，多少归自动整理的缓存架。**

❓ 可能被问："L1 和共享内存的比例谁定？"→ 可以按 kernel 配置（carveout 属性，`cudaFuncSetAttribute` 一行调用）——知道有这个旋钮即可，不展开。
❓ 可能被问："bank 冲突到底是什么？"→ 共享内存物理上分成 32 个 bank（正好对应 warp 的 32 条通道），同一拍里多个线程访问**同一个 bank 的不同地址**就得排队。全 warp 访问连续地址（各落各的 bank）或同一地址（广播）都没事。矩阵转置是经典踩坑场景，第 10 章有解法（padding）。

---

## 第 5 页｜TMEM and TMA: Feeding the Tensor Cores *（20 秒跳过页）

🎤 **这页标了星——Blackwell 的新组件，第 10 章整章细讲，今天 20 秒飞过**，只要记两个名字大概是干啥的：**TMEM**——每 SM 一块 Tensor Core 专属的 SRAM，给矩阵乘当**累加器**，程序员的指针摸不到它；**TMA**——自动搬运工，你给它一张"从哪搬、搬多大"的描述符，它就在后台把数据块搬进搬出。合起来的效果：矩阵乘的中间结果从头到尾**不用碰全局内存**。第 10 章见。（跳下一页）

📖 页面对照：
- 顶部灰字 "* Blackwell's new hardware --- Chapter 10 covers it in depth; a 20-second flyover today." → 星标：Blackwell 新硬件，第 10 章深讲，今天 20 秒飞过。
- TMEM = Tensor Core 的专属累加器 SRAM（不可指针寻址）；TMA = 按描述符自动搬运的引擎；净效果 = 降低对全局内存的依赖。

大白话：**TMA 是在后台运货的叉车班组；大厨（Tensor Core）从头到尾不用离开灶台。**

❓ 若有人追问细节 → "第 10 章有完整的一章，包括 warp specialization 和 descriptor 的写法；今天先记住这两个角色。"

---

## 第 6 页｜Constant Cache: One-Cycle Broadcast（常量缓存：单拍广播）

🎤 这页三句话讲完：**Constant Memory 是一小块全局只读内存**（64 KB 的 `__constant__` 空间），**每个 SM 前面配了一块约 8 KB 的 Constant Cache**。
它的**特殊能力**：如果一个 warp 的 32 个线程**同时读取同一个地址**，只需读取一次，就能把值**广播**给全部 32 个线程——和读寄存器一样快。
但反过来，如果 32 个线程**读取不同地址**，请求会**分批甚至串行**处理。
所以结论看下面两个框：**适合**——小型、只读、**所有线程统一访问**的参数（LLM 里的现成例子：RoPE 查找表、ALiBi 斜率、LayerNorm 的 γ/β、量化 scale）；**不适合**——普通的大 tensor，或每个线程查**不同位置**的表。

📖 页面对照：
- "a small, read-only region of global memory (64 KB __constant__); each SM fronts it with an ~8 KB constant cache" → 一小块全局只读内存，每 SM 前置约 8 KB 缓存。
- "Superpower: same address → one fetch broadcast to all 32 lanes" → 同址读取，一次取回、广播全组。
- "Weakness: different addresses → batched or even serialized" → 不同地址则分批甚至串行。
- Good fit / Bad fit 两框 → 统一访问的小只读参数 ✓；大 tensor、每线程各查各的 ✗。

大白话：**它是喇叭不是信箱：一次广播全班 32 人都听到——前提是大家问的是同一个问题。**

❓ 可能被问："这些数据放共享内存不行吗？"→ 行，但亏：共享内存要自己写代码搬、占 block 的工作台配额；常量缓存的广播是硬件免费送的，而且跨 block 全局有效。判断标准一条：只读 + 全体同址 → 常量内存。

---

## 第 7 页｜L2 Cache: The GPU-Wide Middleman（L2：全 GPU 的中间人）

🎤 往下走一层，出 SM 了。L2 是**全 GPU 唯一一块所有 SM 共享的缓存**：**126 MB**（这个数值得停一秒——比很多 CPU 的 L3 还大），约 200 拍，聚合带宽数 TB/s，同时兜住 L1 放不下的溢出。
它最大的价值是**跨 block 复用**：block A 从 HBM 取过的数据，block B 再要时直接从 L2 拿——**不必再去 DRAM 走一趟**。想想 Part I 的实例：3,907 个 block 处理同一个数组，相邻 block 的数据边界高度重叠，L2 就是它们之间无声的共享层。
右框是书的提示，也是第 7 章的预告：把全局内存访问组织成 **128 字节对齐、可合并（coalesced）**的段，与缓存行整齐对应——避免拆分事务，同时榨满 L2 和 DRAM 带宽。

📖 页面对照：
- "126 MB shared by all SMs --- the glue between on-chip SRAM and off-chip HBM3e" → 全 SM 共享，片上 SRAM 与片外 HBM 之间的胶水层。
- "data fetched by one thread block is reused by others without revisiting DRAM" → 一个 block 取过的数据，别的 block 复用，不再下 DRAM。
- coalescing rule → 128 字节对齐的合并访问（第 7 章细讲）。

大白话：**L2 是全厂的中转仓：别的小组已经从大仓库运来的零件，你直接到中转仓取——前提是大家按"整托盘"下单，别一颗螺丝一颗螺丝地要。**

❓ 可能被问："L2 能编程控制吗？"→ 可以部分控制：CUDA 提供 access policy window（`cudaStreamSetAttribute` 设置 persisting 区域），把某段地址"钉"在 L2 里——推理场景钉 KV cache 是典型用法。进阶话题，知道存在即可。

---

## 第 8 页｜Global HBM3e, the Dual-Die B200, and Coherency（HBM、双芯粒与一致性）

🎤 阶梯的最底层，三个知识点：
① **数字**：B200 上 **180 GB**（B300 约 288 GB），带宽 **约 8 TB/s**——但延迟几百到 1000+ 拍，**整条链路上最慢的一环**。第 3 页说的寄存器溢出、还有超大的自动数组，付的都是这个价。
② **冷知识（书中注）**：B200 物理上是**两块光刻极限大小的裸片**，用 **10 TB/s** 的片间互连拼起来，各带 4 个 HBM3e 栈——但呈现给你的是**一块 GPU、一个地址空间**。10 TB/s 比 HBM 带宽还高，所以你基本感觉不到缝。
③ **一致性点（point of coherency，图 6-15）**：内存一致性按作用域逐级建立——**线程 → block → 簇 → 设备 → 系统**，通信范围越广、代价越高。这解释了 Part I 的设计哲学：为什么协作被限制在 block 内（工作台上一致性最便宜）、为什么跨 block 默认互不通气。
图 6-14 是全局内存的层级位置图。

📖 页面对照：
- "180 GB on B200 at ~8 TB/s --- but hundreds to 1000+ cycles: the slowest link in the chain" → 容量带宽都大，但延迟是全链最慢。
- "two reticle-limited dies joined by a 10 TB/s chip-to-chip interconnect --- presented as one GPU, one address space" → 双芯粒拼接，呈现为一块 GPU。
- "memory consistency is established at thread → block → cluster → device → system scope" → 一致性按作用域逐级建立，越广越贵。

大白话：**大仓库又大又能吞吐（一次能发很多货），但下一单等回货的时间很长——所以要么少下单（片上复用），要么让别的小队在等货时干活（占用率）。**

❓ 可能被问："双芯粒对我写代码有影响吗？"→ 书的态度：当一块 GPU 用，透明。严格说跨芯粒访问对方的 HBM 栈有轻微 NUMA 效应，但 10 TB/s 互连让它基本不可见；除非做极限调优，不用管。

---

## 第 9 页｜Unified Memory Hides Copies, Not Their Cost（统一内存：藏起来的是拷贝，不是代价）⭐

🎤 标题就是这页的全部：**统一内存隐藏的是拷贝这个动作，不是拷贝的代价**。
它是什么：`cudaMallocManaged()` 给你 **CPU+GPU 一个连贯的地址空间**——不用 h_/d_ 两份指针，不用手写 `cudaMemcpy`，Part II 六步流程里的两步拷贝直接消失，代码干净得多。
底层机制（图 6-16）：内存页在 CPU 和 GPU 之间**按需迁移**——谁访问、页搬到谁那边。
代价在哪：GPU 摸到一个还在 CPU 侧的页 → **缺页中断，kernel 当场停摆**，等页搬完才继续。走 PCIe 时，这种按需搬页**可能比手动 memcpy 还慢**（一页一页搬 vs 一大块连续搬）。
硬件缓解：Grace 超级芯片上的 **NVLink-C2C（约 900 GB/s）** 让迁移接近本地速度——**但延迟永远不为零**，"要用时才发现不在"这个结构性问题不变。

📖 页面对照：
- "one coherent CPU+GPU address space --- no separate buffers, no manual cudaMemcpy" → 一个连贯地址空间，免双份缓冲、免手动拷贝。
- "pages migrate on demand; a GPU touch of a CPU-resident page page-faults and stalls the kernel" → 按需迁移；GPU 摸到 CPU 侧的页就缺页停摆。
- "On PCIe: on-fault transfers can be slower than manual memcpy... latency is never zero." → PCIe 上可能更慢；延迟永远不为零。

大白话：**统一内存是自动送料服务——省事，但小组要用时零件还没到车间，全组就只能站着等。**

❓ 可能被问："那生产上到底用不用？"→ 分场景：原型/科研期非常好用（代码量减半）；性能关键路径上要么回到手动管理，要么用下一页的 prefetch + advise 把代价治住。Grace 这类 C2C 平台上实用性大增。

---

## 第 10 页｜Taming Unified Memory: Prefetch, Advise, Attach（驯服统一内存）

🎤 上一页的病，这一页三味药，对应左边三段代码：
**药一：预取**。`cudaMemPrefetchAsync(ptr, size, gpuId, stream)`——**在 kernel 需要之前**把数据搬过去，把"首次触碰才缺页"变成**可与计算重叠的异步传输**（图 6-17）。这是最重要的一味，等价于"提前下单"。
**药二：建议**。`cudaMemAdvise` 三个提示：`SetPreferredLocation`（这块数据的首选住址）；`SetReadMostly`（只读为主 → 允许各处**复制只读副本**，谁读谁有，互不打架）；`SetAccessedBy`（让另一块 GPU **建立映射直接访问而不迁移**——数据不搬家，远程读，图 6-18）。
**药三：绑定**。`cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachSingle)`——把这段内存的缺页/迁移**圈在一条流里**，不让**其他**流被它拖着停。
补充：没有 NVLink-C2C 的平台，把数据钉在 **NUMA 本地**（对等拷贝/预取）。
底线（右栏最后一条）：三味药用齐，性能**接近手动 `cudaMemcpy`**，同时保留统一内存的简洁——鱼和熊掌基本兼得。

📖 页面对照：
- "Prefetch turns first-touch migrations into overlappable async transfers" → 预取把首触迁移变成可重叠的异步传输。
- "ReadMostly allows replicated read-only copies; AccessedBy lets a second GPU map pages without migration" → 只读复制副本；映射访问不迁移。
- "Attach stops other streams from stalling on this range" → 绑定让其他流不被这段内存拖停。
- "performance close to manual cudaMemcpy, simplicity retained" → 性能接近手动拷贝，简洁保留。

大白话：**自动送料没问题，但要提前下单（prefetch）、写清用途（advise）、别让别的班组陪着等（attach）。**

❓ 可能被问："prefetch 和手动 memcpy 还有什么区别？"→ 功能上几乎等价，区别在心智模型：prefetch 是"优化提示"，忘了写程序照样对（只是慢）；memcpy 是"正确性义务"，忘了写就是 bug。统一内存 + prefetch = 默认正确、可选变快。
