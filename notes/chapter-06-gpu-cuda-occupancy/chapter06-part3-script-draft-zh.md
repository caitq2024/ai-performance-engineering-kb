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
右图（图 6-10）是书里的层级图，含 CPU 一侧。这页的任务只是建立地图，后面 8 页逐层展开。

📖 页面对照（表 6-5 各列）：Level 层级 / Scope 作用域 / Capacity 容量 / Latency 延迟 / BW 带宽；"Every level trades capacity for latency/bandwidth." → 每一级都在用容量换延迟/带宽。

大白话：**随身工具箱 → 公共工作台 → 全厂中转仓 → 远处大仓库。活儿尽量在自己工位上干。**

❓ 可能被问："为什么不把所有内存都做成寄存器那么快？"→ 物理规律：越快的存储（SRAM 触发器）单位容量越占面积、越耗电。8 TB/s 的 HBM 已经是堆了十几层 DRAM die 的结果。阶梯不是设计缺陷，是物理约束下的最优解——软件的任务就是把热数据留在上层。

---

## 第 3 页｜Register Pressure Can Turn Fast Storage into DRAM Access（寄存器与溢出悬崖）⭐

🎤 标题就是结论：**寄存器压力能把最快的存储变成 DRAM 访问**。四个要点：
① 每个线程的数据之旅都从**寄存器堆**出发：单拍读写、几乎不与任何人抢，每 SM 几十 TB/s——回忆 Part I：工具箱是从车间大柜子里划走的一格。
② 预算：每 SM 64K 个 32 位寄存器，**单线程最多 255 个**。
③ **悬崖在这**：局部变量太多、编译器临时量太多，超出的部分**溢出（spill）到 local memory**——名字里带"本地"，实际住在**片外 DRAM**：从 1 拍直接掉到几百上千拍，这就是"悬崖"两个字的意思。右图（图 6-12）画的就是这个溢出区。
④ 怎么防：盯住 Nsight Compute 里的 **"Registers Per Thread"** 指标——**溢出是无声的杀手**，代码不报错、结果全对，只是莫名其妙地慢。
和 Part I 的呼应：`__launch_bounds__`（Part IV 会讲）就是主动限制每线程寄存器数来换占用率的工具——那是硬币的另一面。

📖 页面对照：
- "Every thread 'begins its journey' at the register file: single-cycle reads/writes" → 每个线程的数据之旅从寄存器堆开始：单拍读写。
- "overflow spills into local memory --- which despite the name lives in off-chip DRAM" → 溢出进 local memory——名字唬人，实际在片外 DRAM。
- "spills are silent killers" → 溢出是无声的杀手。

大白话：**"local memory" 是 CUDA 里最误导人的名字——它是城另一头的自助仓储，不是你的口袋。**

❓ 可能被问："怎么知道自己 spill 了？"→ 两个办法：编译时 `nvcc -Xptxas -v` 会打印每 kernel 的寄存器数和 spill stores/loads 字节数；运行时 ncu 看 Registers Per Thread 和 local memory 流量。看到 spill 字节数 > 0 就要警惕。

---

## 第 4 页｜Shared Memory + L1: One SRAM, Two Jobs（共享内存与 L1：一块 SRAM 两份工作）

🎤 这页讲清一个容易误解的事实：共享内存和 L1 **物理上是同一块 SRAM**——每 SM 一块 256 KB，一边当**用户手动管理的共享内存**（最多 228 KB，单 block 实际可用 227），一边当**硬件自动管理的 L1 数据缓存**。
分界线（carveout）你可以自己选，就是中间那三行代码：`cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 百分比)`——共享内存用得多的 kernel 就把隔板往共享侧推。
性能画像：约 20–30 拍；想拿到 **TB/s 级**带宽有个前提——避开 **bank 冲突**：多个线程撞上同一个存储 bank，访问就被串行化（细节第 7/10 章）。
用途定位一句话：这是 **block 内协作的工作台**——矩阵分块（tile）、归约、数据中转（staging）都在这干。回忆 Part I：`__syncthreads()` 协调的就是工作台上的交接。

📖 页面对照：
- "One 256 KB SRAM per SM, split between user-managed shared memory and L1/data cache. You choose the carveout." → 一块 SRAM，用户管理的共享内存与 L1 缓存分账，比例你定。
- "TB/s if you avoid bank conflicts (multiple threads hitting the same memory bank → serialized access)" → 避开 bank 冲突才有 TB/s。
- "the workbench for intrablock cooperation --- tiles, reductions, staging" → block 内协作的工作台。

大白话：**一张工作台，隔板可调：多少归你们班组的项目台，多少归自动整理的缓存架。**

❓ 可能被问："bank 冲突到底是什么？"→ 共享内存物理上分成 32 个 bank（正好对应 warp 的 32 条通道），同一拍里多个线程访问**同一个 bank 的不同地址**就得排队。全 warp 访问连续地址（各落各的 bank）或同一地址（广播）都没事。矩阵转置是经典踩坑场景，第 10 章有解法（padding）。

---

## 第 5 页｜TMEM and TMA: Feeding the Tensor Cores（TMEM 与 TMA：给 Tensor Core 喂料）

🎤 这页是 Blackwell 的新硬件，认识两个缩写就够：
**TMEM（Tensor Memory）**：每 SM 又一块**专属的 256 KB SRAM**，专职给第五代 Tensor Core 指令（`tcgen05.*`、UMMA）当**累加器**，与 Tensor Core 之间几十 TB/s。特别之处：**在 CUDA C++ 里不能用指针直接寻址**——你摸不到它。
**TMA（Tensor Memory Accelerator）**：专职搬运工。数据搬运不靠线程一个个搬，而是给 TMA 一张**描述符（descriptor）**——"从哪搬、搬多大的块、什么布局"——它在后台自主执行。
右图（图 6-11）是 C = A × B 的分工：操作数 B 放共享内存，A 和累加器放 TMEM；TMA 把分块沿 **HBM → L2 → SMEM** 搬运；SMEM 和 TMEM 之间由 Tensor Core 指令隐式完成。
净效果一句话：**大幅降低 Tensor Core 对全局内存的依赖**——矩阵乘的中间结果从头到尾不碰 HBM。细节第 10 章，今天只要知道"有这么两个角色"。

📖 页面对照：
- "dedicated 256 KB per-SM SRAM, the accumulator for 5th-gen Tensor Core ops" → 每 SM 专属 SRAM，第五代 Tensor Core 的累加器。
- "Not pointer-addressable from CUDA C++ --- data movement is orchestrated by the TMA via descriptors" → 不可指针寻址；由 TMA 按描述符编排搬运。
- "reduces Tensor Core reliance on global memory" → 降低对全局内存的依赖。

大白话：**TMA 是在后台运货的叉车班组；大厨（Tensor Core）从头到尾不用离开灶台。**

❓ 可能被问："TMEM 和共享内存什么区别？"→ 共享内存是通用工作台，程序员指针可达；TMEM 是 Tensor Core 的专用累加器，只有 Tensor Core 指令能读写。这样设计是为了让矩阵乘的累加带宽不去挤共享内存的通道。

---

## 第 6 页｜Constant Cache: One-Cycle Broadcast（常量缓存：单拍广播）

🎤 最小但最有性格的一层。硬件：每 SM 约 **8 KB 缓存**，服务 64 KB 的 `__constant__` 地址空间。
它的**超能力**：当一个 warp 的 32 个线程读**同一个地址**时，缓存**一拍把值广播给全部 32 条通道**——和读寄存器一样快。
它的**脾气**：32 个线程读**不同**地址就退化——按通道串行化；缓存未命中代价更高。所以使用条件三个词：**小的、只读的、访问一致的**数据。
右框是书里给的 LLM 实例，都很实在：**RoPE 位置编码查找表、ALiBi 斜率、LayerNorm 的 γ/β 向量、量化 scale 系数**——这些都是全体线程反复读同一份的小数据，放常量内存后**全局内存流量为零**。

📖 页面对照：
- "when all 32 threads of a warp load the same address, the cache broadcasts the value in a single cycle --- as fast as a register" → 全 warp 同址读取，一拍广播，与寄存器同速。
- "Divergent reads serialize across lanes" → 各读各的就按通道串行。
- "small, read-only, uniformly-accessed data" → 小、只读、访问一致。

大白话：**它是喇叭不是信箱：一次广播全班 32 人都听到——前提是大家问的是同一个问题。**

❓ 可能被问："这些数据放共享内存不行吗？"→ 行，但亏：共享内存要自己写代码搬进去、占掉 block 的工作台配额；常量缓存的广播是硬件免费送的，而且跨 block 全局有效。判断标准就一条：只读 + 全体同址 → 常量内存。

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
