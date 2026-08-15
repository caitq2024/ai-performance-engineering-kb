# 第六章讲稿（中英对照，逐页对应 chapter06-presentation.pdf）

> **用法**：左边开着 `chapter06-presentation.pdf`，右边看这份讲稿。每一节对应一页幻灯片，包含四块内容：
> - 🎤 **讲稿**——可以直接照着念的中文口播稿（英文术语第一次出现时带中文解释）；
> - 📖 **页面英文对照**——幻灯片上英文内容的翻译，保证你知道自己屏幕上每句话是什么意思；
> - 📚 **原书精读**——书里最重要的英文原句 + 逐句翻译（读完这些，第六章的核心原文你就都过了一遍）；
> - ❓ **可能被问**——预判提问和参考回答（不是每页都有）。
>
> **建议时长**：整场 35–45 分钟。每页大约 1.5 分钟，重点页（第 5、8、13、17、20 页）可以讲 3 分钟。

---

## 第 1 页｜标题页

🎤 **讲稿**：
大家好，今天我来讲第六章：GPU Architecture, CUDA Programming, and Maximizing Occupancy——GPU 架构、CUDA 编程和最大化"占用率"。前面几章讲的是系统层面：硬件选型、网络、存储。从这一章开始，全书进入 GPU 内部，讲怎么写出高效的 CUDA 代码。这一章是后面第 7 到 12 章所有 kernel 级优化的**地基**，所以概念会多一点，但每个概念我都会用大白话解释。

📖 **术语对照**（这几个词全章反复出现，先混个脸熟）：
| 英文 | 中文 | 一句话解释 |
|---|---|---|
| Occupancy | 占用率 | GPU 的"工位坐满率"：活跃 warp 数 ÷ 硬件上限 |
| SM (Streaming Multiprocessor) | 流式多处理器 | GPU 的"车间"，类比 CPU 的一个核 |
| Warp | 线程束 | 32 个线程绑在一起同步执行的最小调度单位 |
| Kernel | 核函数 | 跑在 GPU 上的函数 |
| Latency hiding | 延迟隐藏 | 一个 warp 等内存时，切换到别的 warp 干活 |

---

## 第 2 页｜What This Talk Is About — and Why（这次分享讲什么、为什么重要）

🎤 **讲稿**：
先用一句大白话概括整章：**GPU 是一台"吞吐量机器"**。它不追求单个线程跑得快，而是同时跑几千个线程，用"总有人在干活"来掩盖内存慢的问题。这一章教三样东西：第一，**词汇表**——warp、block、grid、SM 这些概念到底是什么；第二，**内存的梯子**——从寄存器到共享内存到 L2 到 HBM，一层比一层大、也一层比一层慢；第三，**一条黄金法则**：先启动足够多的并行工作，把每个 SM 喂饱。
为什么重要？因为一块 GPU 如果 SM 都在空转，那就是一台很贵的电暖器——很多没调优的 kernel 只用到了硬件的百分之几。而 occupancy（占用率）就是衡量"喂饱程度"的指标，roofline（屋顶线）模型则是"指南针"，告诉你该优化计算还是优化访存，避免优化错方向。

📖 **页面英文对照**：
- "A GPU is a **throughput machine**: it runs **thousands of threads** at once and hides slow memory by always having other work ready."
  → GPU 是吞吐量机器：同时跑几千个线程，靠"手里永远有别的活"来掩盖内存慢。
- "The problem: a GPU with idle SMs is an expensive space heater."
  → 问题：SM 空转的 GPU 就是一台昂贵的取暖器。
- "The compass: the roofline model tells you if a kernel is compute-bound or memory-bound **before you optimize the wrong thing**."
  → 指南针：roofline 模型在你"优化错东西"之前，先告诉你 kernel 是算力受限还是访存受限。

📚 **原书精读**（本章开篇）：
> "Unlike CPUs, which optimize for low-latency single-thread performance, GPUs are throughput-optimized processors built to run thousands of threads in parallel."
> 与优化单线程低延迟的 CPU 不同，GPU 是为并行运行几千个线程而生的**吞吐量优化型**处理器。

---

## 第 3 页｜Roadmap（路线图）

🎤 **讲稿**：
今天按这六步走：一，GPU 架构——SM 里面长什么样、线程怎么组织、什么是 warp 分叉；二，CUDA 编程速览——一个 kernel 的骨架、启动参数怎么选、内存怎么异步分配；三，内存层级——从寄存器到 HBM 的完整梯子，包括 Blackwell 新增的 TMEM；四，occupancy——为什么"并行度是第一法则"，我会给一个 22 倍加速的案例；五，正确性——十万个线程的程序怎么查 bug；六，roofline 模型——判断瓶颈在哪、对症下药。
一句话记住主线：**先让 GPU 忙起来，再让每个时钟周期都花在刀刃上，全程用 profiler 说话。**

📖 **页面英文对照**：页面底部的斜体 "Plain English" 是原书的特色写法，意思是"说人话版本"。这一页的说人话版本是："first make the GPU busy, then make each cycle count, and always profile to know which fix applies"——先让 GPU 忙起来，再抠每个周期，永远用 profiling 决定用哪个药方。

---

## 第 4 页｜GPUs Optimize Throughput; CPUs Optimize Latency（GPU 重吞吐，CPU 重延迟）

🎤 **讲稿**：
先看最基本的 CPU-GPU 协作流程，就是右边这张图（Figure 6-1）：主机（host，就是 CPU）先把数据从 CPU 内存拷到 GPU 显存，然后启动 kernel，算完再把结果拷回来。注意这里有两次跨设备拷贝，都不便宜。GPU 应对的办法不是把拷贝变快，而是**用海量并行把延迟藏起来**——一个 warp 在等数据的时候，调度器立刻切到另一个准备好的 warp 去执行。
量级感受一下：一块 Blackwell GPU 有一百多个 SM，每个 SM 能同时"挂"64 个 warp，也就是 2048 个线程。所以一块卡上几十万个线程同时在册，是常态。
大白话总结：**CPU 是跑车，GPU 是货运列车——别要求列车快，要让列车装满。**

📖 **页面英文对照**：
- "CPUs: few cores, deep caches, fast *single* threads. GPUs: hundreds of SMs running thousands of threads in parallel."
  → CPU：核少、缓存深、单线程快。GPU：几百个 SM 并行跑几千个线程。
- "Each Blackwell SM tracks up to **64 warps** (2,048 threads) concurrently."
  → 每个 Blackwell SM 可同时跟踪 64 个 warp（2048 个线程）。

📚 **原书精读**：
> "GPUs rely on massive parallelism to hide data-transfer latency."
> GPU 依靠海量并行来隐藏数据传输延迟。
> "Each GPU comprises many SMs, which are roughly analogous to CPU cores but streamlined for parallelism."
> 每块 GPU 包含许多 SM——大致类比 CPU 的核，但为并行做了精简。

---

## 第 5 页｜Inside a Blackwell SM: Four Schedulers, Dual Issue（SM 内部：四个调度器、双发射）⭐重点页

🎤 **讲稿**：
现在打开一个 SM 看内部，就是右边这张图（Figure 6-2）。每个 SM 其实是**四个"迷你 SM"**：四个独立的 warp scheduler（warp 调度器），各管一片执行单元，共享片上资源。
两个关键机制：第一，每个调度器每个时钟周期可以发射一个 warp 的指令，四个调度器加起来，**每拍最多让 4 个 warp 同时往前走**。第二，叫 **dual-issue，双发射**——同一个 warp 在同一拍里可以同时发出两条指令：一条算术指令（比如 INT32、FP32 或 Tensor Core 运算）加一条访存指令（load 或 store）。注意限制：这两条指令必须来自**同一个 warp**，不能跨 warp 拼。所以最好情况是每拍 4 条数学指令加 4 条访存指令同时飞。
还有一个容易忽略的角色：**SFU，Special Function Unit（特殊功能单元）**，专门算 sin、cos、开方、倒数这类超越函数。它有自己独立的管线，不占双发射的名额——所以偶尔算个 sin 不会把主管线堵住。
资源数字记两个就够：每个 SM 有 **64K 个 32 位寄存器**（共 256 KB），和 **256 KB 的统一 L1/共享内存**。
大白话：**一个 SM 是四条收银通道，每个收银员每拍扫一个客户的商品，而且可以一只手扫码（算术）、另一只手装袋（访存）。**

📖 **页面英文对照**：
- "Each SM = four 'mini-SMs': independent warp schedulers sharing on-chip resources."
  → 每个 SM = 四个"迷你 SM"：独立的 warp 调度器，共享片上资源。
- "each scheduler ... can **dual-issue** one math + one memory op *from the same warp*."
  → 每个调度器可以双发射：同一 warp 的一条算术 + 一条访存。
- "SFUs (sin, cos, sqrt, ...) run in a separate pipeline — transcendentals don't stall the core pipes."
  → SFU 走独立管线——超越函数不会堵住核心管线。

📚 **原书精读**：
> "You can think of the SM as four 'mini-SMs' sharing on-chip resources. This lets the hardware pick ready warps and issue instructions from up to four different warps each clock cycle."
> 可以把 SM 想成四个共享片上资源的"迷你 SM"。这让硬件每个时钟周期能从最多四个不同的 warp 中挑选就绪的发射指令。
> "Note that the dual-issue must come from the same warp—and not across warps."
> 注意：双发射的两条指令必须来自同一个 warp，不能跨 warp。

❓ **可能被问**："每个调度器有几条访存管线？" → 书里说每个调度器约 4 条 LD/ST（load/store）管线、全 SM 共 16 条，但作者特别提醒：**具体数目和配对规则不保证**（"Exact LD/ST pipeline counts and pairings are not guaranteed"），应以 profiling 和 NVIDIA 官方文档（Blackwell tuning guide）为准。

---

## 第 6 页｜The Thread Hierarchy: Threads → Blocks → Grids（线程层级）

🎤 **讲稿**：
CUDA 把并行工作组织成三层，看右图（Figure 6-3）。最底层是 **thread（线程）**——执行你 kernel 代码的一个工人，处理一个数据元素。往上是 **thread block（线程块）**，书里也叫 CTA（cooperative thread array），最多 1024 个线程一组；同一 block 内的线程可以用超快的 **shared memory（共享内存）** 交换数据，还能用 `__syncthreads()` 同步——但每次同步是有开销的，所以书里强调**尽量少设同步点**。最上层是 **grid（网格）**：一次 kernel 启动的全部 block。
最关键的设计哲学在最后一条：**block 之间互相独立、执行顺序不保证任何东西**。正因为这个约束，GPU 调度器才能把 block 随意撒到任何 SM 上；也正因为这个约束，你今天写的代码在未来 SM 更多的 GPU 上**不改一行**就能自动扩展。
大白话：**工人（线程）组成班组（block），班组内可以低成本交流；公司（grid）按活儿多少雇任意多个班组。**
顺带提一句书里的一个新特性：现代架构支持 **thread block cluster（线程块簇）**——多个 block 可以跨 SM 互访共享内存，底层靠 **DSMEM（分布式共享内存）** 硬件支持。这个第 10 章才细讲，今天知道有这么个东西就行。

📖 **页面英文对照**：
- "Blocks execute **independently, in any order** — that freedom is what lets the same code run on future GPUs with more SMs."
  → block 独立执行、顺序任意——正是这个自由度让同一份代码能跑在未来 SM 更多的 GPU 上。

📚 **原书精读**：
> "By sizing your grid appropriately, you can scale to millions of threads without changing your kernel logic."
> 只要把 grid 的尺寸设对，你可以扩展到几百万个线程，而不用改任何 kernel 逻辑。
> "Because each barrier incurs overhead, you should minimize synchronization points."
> 因为每个同步屏障都有开销，应该尽量减少同步点。

---

## 第 7 页｜Warps and SIMT: 32 Threads in Lockstep（Warp 与 SIMT：32 个线程齐步走）

🎤 **讲稿**：
block 再往下切，就到了硬件真正的调度单位：**warp**，固定 32 个线程。这 32 个线程在 **SIMT** 模型下执行——Single Instruction, Multiple Threads，**同一条指令，多个线程一起执行**。大白话：**warp 是 32 人的划船队，所有人同一拍划同一个动作，谁也不能自己划自己的。**
现在可以正式定义本章的标题概念了：**occupancy（占用率）= SM 上实际活跃的 warp 数量占硬件上限（64）的比例**。为什么追求高 occupancy？因为 warp 等内存的时候，调度器可以零成本切换到另一个 warp——**在飞的 warp 越多，延迟藏得越好**。
但马上要说"但是"：occupancy 不能无脑拉满。每个线程用的寄存器、每个 block 用的共享内存都是有限资源——warp 塞太多，每个线程分到的寄存器就少；寄存器不够用会发生 **register spilling（寄存器溢出）**，数据被挤到慢几百倍的显存里去，反而制造了新的停顿。这是全章最重要的权衡，第 18 页会给出调法。

📖 **页面英文对照**：
- "High occupancy ⇒ when one warp stalls on memory, another is ready ⇒ **latency hiding**."
  → 高占用率 ⇒ 一个 warp 卡在访存上时另一个随时顶上 ⇒ 延迟隐藏。
- "too many registers/shared memory per thread ⇒ fewer resident warps, or **register spilling** to slow memory."
  → 每线程寄存器/共享内存用太多 ⇒ 常驻 warp 变少，或者寄存器溢出到慢速内存。

📚 **原书精读**：
> "Keeping more warps in flight is known as high occupancy on the SM. When your CUDA code allows high occupancy, it means that when one warp stalls, another is ready to run."
> 让更多 warp "在飞"就叫高占用率。代码允许高占用率时，一个 warp 停下来，另一个马上能跑。

---

## 第 8 页｜Warp Divergence（Warp 分叉）⭐重点页

🎤 **讲稿**：
SIMT 的"齐步走"有一个天生的软肋：**分支**。看右图（Figure 6-8）。如果**同一个 warp 里**有的线程满足 `if`、有的走 `else`，硬件没法让 32 个人同时走两条路——它只能**串行化**：先执行 if 分支，把走 else 的那些线程的"车道"（lane）**屏蔽掉**（masked）；再执行 else 分支，屏蔽走 if 的。这叫 **warp divergence（warp 分叉）**，执行时间直接**乘以分支路径的数量**。
两个要点必须说清楚：第一，这只发生在**一个 warp 内部**——**不同 warp 走不同分支完全没有代价**，各划各的船互不影响。第二，怎么发现和治理是第 8 章的内容，今天只要能识别这个现象。
大白话：**如果划船队里一半人往左划、一半人往右划，船就得把两个动作各做一遍——慢一倍。**

📖 **页面英文对照**：
- "the warp **serializes**: it runs the `if` path with half the lanes masked, then the `else` path."
  → warp 串行化：先蒙住一半"车道"执行 if 路径，再执行 else 路径。
- "**No penalty across warps** — different warps may branch differently for free."
  → 跨 warp 无惩罚——不同 warp 各走各的分支，免费。

📚 **原书精读**：
> "By masking inactive lanes and running extra passes to cover each branch, warp divergence multiplies the overall execution time by the number of branches."
> 通过屏蔽不活跃车道、跑多遍来覆盖每个分支，warp 分叉会让总执行时间乘以分支数。
> "Divergence is an issue only for threads within a single warp. Different warps can follow different branches with no performance penalty."
> 分叉只是**单个 warp 内**线程的问题。不同 warp 走不同分支没有性能惩罚。

❓ **可能被问**："那 threads 按什么规则分进 warp？" → 按线程编号连续切：threadIdx 0–31 是第一个 warp，32–63 是第二个，以此类推。所以写分支条件时尽量让相邻 32 个线程走同一条路（比如按 `threadIdx.x / 32` 分支就无害，按 `threadIdx.x % 2` 分支就是最坏情况）。

---

## 第 9 页｜Hardware Limits That Shape Your Launch（决定启动配置的硬件上限）

🎤 **讲稿**：
这一页是"查表页"，讲的时候不用逐行念，挑三个数说：
第一，**warp 固定 32 线程**——所以 block 尺寸永远选 32 的倍数。反例：33 线程的 block 要占**两个** warp 槽位，第二个 warp 只有 1 个线程干活、31 个空转，但它照样占一个调度器名额。
第二，**每个 block 最多 1024 线程**（等于 32 个 warp）。
第三，**每个 SM 的常驻上限：64 个 warp、2048 个线程、32 个 block**。这组数字决定了 occupancy 的天花板：如果每个 block 用 1024 线程，一个 SM 最多驻 2 个 block；改成 256 线程的 block，就能驻 8 个——block 小一点反而更灵活。
grid 的上限（X 维 21 亿个 block）基本用不完——实际中你**永远先撞上 per-SM 的限制**。

📖 **页面英文对照**（表格行）：
| 英文 | 中文 |
|---|---|
| Warp size: 32 threads | warp 尺寸：32 线程 |
| Max threads / block: 1,024 | 每 block 最大线程数：1024 |
| Max resident warps / SM: 64 | 每 SM 最大常驻 warp：64 |
| Max resident blocks / SM: 32 | 每 SM 最大常驻 block：32 |
| Registers / SM: 64K × 32-bit (255 / thread) | 每 SM 寄存器：64K 个 32 位（单线程上限 255 个）|
| Shared memory / SM: 228 KB (227 usable) | 每 SM 共享内存：228 KB（可用 227，CUDA 保留 1 KB）|

📚 **原书精读**：
> "Using smaller blocks (e.g., 256 threads) allows more blocks to reside on the SM (up to 8 blocks × 256 = 2,048 threads), which can increase occupancy and help hide latency—though too many tiny blocks can add scheduling overhead."
> 用较小的 block（如 256 线程）能让更多 block 驻留在 SM 上（最多 8×256=2048 线程），提高占用率、帮助藏延迟——但过多的迷你 block 会增加调度开销。

❓ **可能被问**："这些数字换代会变吗？" → 64 warps/SM 这个上限已经保持了好几代（书里原话 "This limit has held for many generations"），但寄存器、共享内存等具体数字每代不同，要查对应架构的 spec。另外书里有个免责声明：**所有指标表的数值是示意性的**（illustrative），真实 benchmark 在书的 GitHub 仓库。

---

## 第 10 页｜Anatomy of a CUDA Kernel（一个 CUDA Kernel 的解剖）⭐重点页

🎤 **讲稿**：
左边是全书第一段完整的 CUDA 代码，五个零件挨个看：
1. **`__global__`**：告诉编译器这个函数跑在 GPU（device）上、从 CPU（host）调用。
2. **`<<<blocksPerGrid, threadsPerBlock>>>`**：三尖括号是 CUDA 特有的启动语法，两个参数就是"雇多少个班组、每班组多少人"。
3. **`int idx = blockIdx.x * blockDim.x + threadIdx.x`**：全 kernel 最重要的一行。三个内置变量——blockIdx 是"我在第几个班组"、blockDim 是"每班组多少人"、threadIdx 是"我是班组里第几号"——拼出一个**全局唯一编号**，决定这个线程处理数组的哪个元素。
4. **`if (idx < N)` 边界检查**：为什么必须有？举书里的例子：N=63，调度器会派两个 warp（64 线程）来。第二个 warp 的第 64 号线程如果不检查就会访问 `input[63]` 之外的地址，直接报 `cudaErrorIllegalAddress`（非法地址错误）。
5. **错误是"懒惰上报"的**：kernel 异步执行，越界不会当场抛异常，而是设一个全局故障标志，等你下次调用同步或其他 CUDA API 时才冒出来。所以规范写法是启动后紧跟 `cudaGetLastError()` + `cudaDeviceSynchronize()` 主动查错。
最核心的思维转变：**你不是在写"处理整个数组的函数"，而是在写"一个工人的作业说明书"——CUDA 把它复印一百万份，每份发一个不同的行号。** 这也回答了"为什么要传 N"：kernel 是单线程视角，N 划定了它的工作边界。

📖 **页面英文对照**：
- "The kernel is written for **one thread**; `idx` carves out its slice of the data."
  → kernel 是按"一个线程"的视角写的；idx 切出属于它的那片数据。
- "Errors surface **lazily** — check `cudaGetLastError()` after launches."
  → 错误是延迟浮现的——启动后要用 cudaGetLastError() 查。

📚 **原书精读**：
> "a CUDA kernel function is designed to work inside of a single thread, alongside thousands of other threads, on a partition of the input data."
> CUDA kernel 函数的设计就是在单个线程内工作、与几千个其他线程并肩、各自处理输入数据的一个分区。
> "You will see a bounds check in a lot of CUDA kernels. If you don't see it, you should understand why it's not there."
> 你会在大量 CUDA kernel 里看到边界检查。如果没看到，你应该搞清楚它为什么不在。

---

## 第 11 页｜Choosing Launch Parameters（启动参数怎么选）

🎤 **讲稿**：
上一页的 256 和那个除法公式不是随便写的，这一页给"食谱"。
**threadsPerBlock 从 256 开始**，四个理由：① 它是 32 的倍数，没有半空的 warp；② 一个 SM 要几百个线程才能藏住 DRAM 延迟，8 个 256 线程的 block 正好填满 2048 的容量；③ 256 线程 = 8 个 warp，通常不会把寄存器和共享内存用爆；④ 它离 1024 的上限很远，调整余地大。书里给 Blackwell 的建议区间是 **256–512**。
**blocksPerGrid 用公式 `(N + threadsPerBlock - 1) / threadsPerBlock`**——这是"向上取整"的标准写法，保证 N 不是 256 整数倍时最后的尾巴也有人管（配合边界检查）。
二维数据同理：比如 1024×1024 的图像，用 16×16 的 block（一样是 256 线程），语法上用 `dim3` 类型，边界检查变成 `if (x < width && y < height)`。
大白话：**256 是 block 尺寸里的"中杯咖啡"——几乎不会点错，之后按 profiling 微调。**

📚 **原书精读**：
> "Starting with threadsPerBlock=256, you can tune up or down (128, 512, etc.) based on your kernel's register and shared-memory requirements—as well as occupancy characteristics."
> 从 256 开始，根据 kernel 的寄存器、共享内存需求和占用率特性上下调整（128、512 等）。

---

## 第 12 页｜Allocate Asynchronously: Streams and Memory Pools（异步分配：流与内存池）

🎤 **讲稿**：
前面例子里用的 `cudaMalloc`/`cudaFree` 有个隐藏成本：它们是**同步的、而且贵**——每次调用都要做一次全设备同步，还要走操作系统的 mmap/ioctl 调用，有内核态切换。偶尔分配一次无所谓；但训练循环里每轮都分配释放，这个开销就积少成多。
解法是右上代码的三步：先建一个 **non-blocking stream（非阻塞流）**——stream 可以理解为 GPU 上的一条"传送带"，同一条带上的操作按序执行，不同带互不干扰；然后用 **`cudaMallocAsync`/`cudaFreeAsync`** 在这条带上分配和释放。它们底层用**内存池（memory pool）**：释放的内存回池子里等复用，不再找操作系统要新的——既省了系统调用，又减少了**碎片化（fragmentation）**。注意 `cudaFreeAsync` 只等自己这条 stream 的活干完，**不会**触发全局同步。
跟大家日常最相关的一句：**PyTorch 的 caching allocator 干的就是这件事**——环境变量 `PYTORCH_ALLOC_CONF`（旧名 PYTORCH_CUDA_ALLOC_CONF）配置的那个分配器，避免每建一个 tensor 都调一次昂贵的 cudaMalloc。
大白话：**别每次送货都"买一辆卡车、用完再卖掉"（手续费吓人）——在停车场养一支车队，用的时候拿钥匙就走。**

📖 **页面英文对照**：
- "`cudaMalloc`/`cudaFree` are **synchronous and expensive**: full-device sync + OS calls."
  → cudaMalloc/cudaFree 是同步且昂贵的：全设备同步 + 操作系统调用。
- "Non-blocking streams avoid legacy **default-stream barriers**."
  → 非阻塞流避开了老式"默认流"的隐式同步屏障。（默认流 stream 0 有个历史包袱：它会跟其他流互相等，用 cudaStreamNonBlocking 建的流没有这个问题。）

📚 **原书精读**：
> "A memory pool recycles freed memory buffers and avoids repeated OS calls to allocate new memory."
> 内存池回收已释放的缓冲区，避免反复找操作系统分配新内存。

---

## 第 13 页｜The GPU Memory Hierarchy（GPU 内存层级）⭐重点页，建议讲 3 分钟

🎤 **讲稿**：
这是全章的核心表格，从上往下，**容量越来越大、速度越来越慢**：
- **寄存器（Registers）**：每线程私有，读写约 1 个时钟周期、基本免费，带宽几十 TB/s。但每线程最多 255 个——用超了就**溢出（spill）**到表格倒数第二行的 local memory，那可是几百上千个周期的 DRAM 速度，性能悬崖。
- **共享内存 + L1**：每个 SM 256 KB 的片上 SRAM，可配置最多 228 KB 当用户管理的共享内存。延迟 20–30 周期，避开 bank conflict（存储体冲突）能到 TB/s 级。这是 block 内线程协作的主战场。
- **TMEM**：Blackwell 新增，下一页专讲。
- **常量缓存（Constant cache）**：8 KB 的小缓存，前置 64 KB 的 `__constant__` 空间。绝活是**广播**：一个 warp 的 32 个线程读同一个地址时，1 个周期广播给所有人，跟读寄存器一样快。适合放小查找表——书里点名了 RoPE（旋转位置编码）表、ALiBi 斜率、LayerNorm 的 γ/β 这些 LLM 里的小常量。
- **L2**：全 GPU 共享的 **126 MB**，约 200 周期。它是所有 SM 通向 HBM 的中转站——一个 block 取过的数据，别的 block 能从 L2 复用，不用再下 DRAM。
- **HBM3e（全局显存）**：B200 是 180 GB、约 8 TB/s，但延迟几百到一千周期——容量最大、也是最慢的一环。
两条行动准则：**能复用就往上层放**（寄存器/共享内存/L2）；**必须下 HBM 时要 coalesced（合并访存）**——把访问组织成 128 字节对齐的连续段，正好映射一条缓存线，避免一次访问被拆成多个事务。
大白话：**书桌（寄存器）→ 办公室书架（共享内存）→ 楼里的图书馆（L2）→ 城另一头的仓库（HBM）。尽量在书桌上干活，去仓库就一次搬一整箱。**
右下角冷知识：B200 物理上是**两块 die**（芯片裸片）用 10 TB/s 的片间互连拼起来的，各接 4 个 HBM 栈;但对开发者呈现为一个统一地址空间，不用特殊处理。

📖 **页面英文对照**（表头）：Level=层级，Scope=作用域，Capacity=容量，Latency=延迟，BW=带宽。cyc = cycle（时钟周期），bcast = broadcast（广播）。

📚 **原书精读**：
> "Here, you can see why maximizing data reuse in registers, shared memory, and L1/L2 cache—and minimizing reliance on global memory—is essential for high-throughput GPU kernels."
> 由此可见：把数据复用最大化地留在寄存器、共享内存和 L1/L2 缓存里、把对全局显存的依赖降到最低，对高吞吐 kernel 是决定性的。

---

## 第 14 页｜TMEM and TMA: Feeding the Tensor Cores（TMEM 与 TMA：喂饱 Tensor Core）

🎤 **讲稿**：
Blackwell 给每个 SM 加了一块 **256 KB 的专用 SRAM，叫 TMEM（Tensor Memory）**，专门服务第五代 Tensor Core 指令（书里写作 tcgen05，配套的矩阵乘累加指令叫 **UMMA**——unified matrix-multiply-accumulate）。它给 Tensor Core 提供几十 TB/s 的带宽，最重要的角色是当**累加器（accumulator）**。
特别之处：**TMEM 不是你能用指针访问的内存**——CUDA C++ 里拿不到它的地址。数据进出全由 **TMA（Tensor Memory Accelerator，张量内存加速器）** 按"描述符"自动搬运：HBM ↔ 共享内存 ↔ TMEM。看右图（Figure 6-11）的矩阵乘 C = A×B：操作数 B 在共享内存，A 和累加器在 TMEM，数据块（tile）由 TMA 从 HBM 经 L2 流进来。
今天只需要记两点：**它存在**，而且它**大幅减少了 Tensor Core 对全局显存的依赖**——这正是上一页"数据往上层放"原则的硬件化。细节在第 10 章。
大白话：**TMA 是专职叉车队，在后台不停搬运物料，让做矩阵乘的大厨永远不用离开厨房。**

📚 **原书精读**：
> "TMEM is a dedicated ~256 KB per-SM on-chip memory used by Blackwell's 5th-generation Tensor Core instructions. It isn't directly pointer-addressable from CUDA C++."
> TMEM 是每 SM 约 256 KB 的专用片上内存，供 Blackwell 第五代 Tensor Core 指令使用。它不能从 CUDA C++ 里用指针直接寻址。

---

## 第 15 页｜Unified Memory（统一内存：一个地址空间、看不见的迁移）

🎤 **讲稿**：
**Unified Memory（统一内存，又叫 CUDA Managed Memory）**：用 `cudaMallocManaged()` 分配，CPU 和 GPU 共享一个地址空间，页面**按需自动迁移**——你再也不用手写 cudaMemcpy，非常省心。
但省心有价：如果 GPU 线程碰到一个还躺在 CPU 内存里的页，GPU 会**缺页中断（page fault）并停下来等**这个页搬过来。在 PCIe 时代这种迁移可能比手动 memcpy 还慢；在 Grace Blackwell 这种超级芯片上，CPU-GPU 之间是 **NVLink-C2C**、约 900 GB/s，迁移接近设备原生速度——但延迟仍然不是零，kernel 中途"意外缺页"照样卡顿。
治理"意外"的三板斧（右边代码）：
1. **`cudaMemPrefetchAsync`**——启动 kernel **之前**把数据整体预取到目标设备，把"第一次摸数据就缺页"变成可重叠的异步传输；
2. **`cudaMemAdvise`** 给驱动递小抄：`SetPreferredLocation` 说"这数据主要在这儿用"，`SetReadMostly` 说"基本只读"（驱动可以在两边各放一份副本），`SetAccessedBy` 让另一块 GPU 直接映射而不触发迁移；
3. **`cudaStreamAttachMemAsync`** 把一段内存绑定到一条 stream，别的 stream 就不会因为它意外停顿。
大白话：**统一内存是客房服务——很方便，但不提前点单（prefetch），开饭的时候就得饿着等。**

📚 **原书精读**：
> "any unexpected page-fault during a kernel launch will stall the GPU while the runtime moves the needed page into place."
> kernel 执行中任何意外的缺页都会让 GPU 停下来，等运行时把所需的页挪到位。
> "With techniques like proactive prefetching, targeted memory advice, and stream attachment, Unified Memory can deliver performance very close to manual cudaMemcpy while preserving the simplicity of a unified address space."
> 用上主动预取、定向内存建议和 stream 绑定这些手段，统一内存能在保住"单一地址空间"简洁性的同时，达到非常接近手动 cudaMemcpy 的性能。

---

## 第 16 页｜Occupancy Case Study: One Thread vs. Many（占用率案例：单线程 vs 多线程）

🎤 **讲稿**：
进入全章的高潮案例：同一个任务——两个百万元素的向量相加 C = A + B——两种写法。
左边 **addSequential**：只有 `blockIdx.x == 0 && threadIdx.x == 0` 的那**一个线程**干活，for 循环加完一百万个元素，其他所有线程、所有 SM 全程围观。启动配置就是 `<<<1,1>>>`。
右边 **addParallel**：每个线程加**一个**元素，`<<<(N+255)/256, 256>>>` 启动约 3907 个 block、一百万个线程同时开工。
左下角这句话请务必带到：**同样的错误在 PyTorch 里更常见也更隐蔽**——用 Python 的 for 循环逐元素写 `C[i] = A[i] + B[i]`，等于往 GPU 上串行发射一百万个迷你 kernel；而正确写法就一行：`C = A + B`，一个向量化 kernel 全并行。书里的忠告：除非你在写全新的东西，几乎总有现成的 PyTorch 原生向量化实现，**不要在 GPU 操作外面套 Python 循环**。
大白话：**GPU 上最常见的性能 bug 不是"kernel 写慢了"，而是"不小心把 GPU 当成了一台很贵的单核 CPU"。**

📚 **原书精读**：
> "In this single-threaded version, the GPU's vast resources are mostly idle. Only one warp, or even one thread within the warp, is doing work while all others sit idle."
> 在单线程版本里，GPU 的庞大资源基本闲置。只有一个 warp——甚至只有 warp 里的一个线程——在干活，其余全部空转。

---

## 第 17 页｜Case Study Results: 22× from Occupancy Alone（结果：光靠占用率就是 22 倍）⭐重点页

🎤 **讲稿**：
用 Nsight Systems 和 Nsight Compute 实测，左边表格四行数据（书里注明是示意值，真实数据在配套 GitHub 仓库）：
- **kernel 时间：48.21 ms → 2.17 ms，22 倍**；
- **GPU 利用率：1.5% → 95%**；
- **achieved occupancy（实际达成的占用率）：1.3% → 38.7%**——注意，38.7% 就够拿到 22 倍了，根本没到 100%；
- **warp execution efficiency（warp 执行效率）：3.1% → 100%**——这个指标说的是"warp 里 32 个车道有多少在干有用的活"，串行版一个 warp 只有 1 个线程干活，所以是 1/32 ≈ 3.1%。
右图（Figure 6-20）解释了加速从哪来：串行版的时间线是"一长条操作 + 等内存的空档"；并行版里这些空档**被其他 warp 的工作填满了**——这就是 latency hiding 的可视化。
最后必须泼一盆冷水（这也是本章反复强调的）：**occupancy 是必要条件，不是充分条件**。如果 kernel 本身是 memory-bound（访存受限），占用率拉到 100% 也没用——瓶颈在带宽。书里给的现实例子就是 **LLM 的 decode 阶段**：逐 token 生成时要把几百 GB 的权重从 HBM 反复搬进片上，光靠加线程救不了它。这就自然引出后面的 roofline。

📖 **页面英文对照**：
- "Rule #1 of CUDA performance: **launch enough parallel work**."
  → CUDA 性能第一法则：启动足够多的并行工作。
- "at high occupancy with a **memory-bound** kernel, more warps won't help — the bottleneck is bandwidth."
  → 对访存受限的 kernel，占用率再高、warp 再多也没用——瓶颈是带宽。

📚 **原书精读**：
> "No matter how fast each thread is, you need lots of threads to leverage the GPU's throughput potential."
> 单个线程再快也没用，你需要**大量**线程才能榨出 GPU 的吞吐潜力。
> "GPU FLOPS are outpacing memory bandwidth... optimizing memory movement is absolutely critical to avoid memory-bound bottlenecks in modern AI workloads."
> GPU 算力的增速正在甩开显存带宽……在现代 AI 工作负载中，优化数据搬运对避免访存瓶颈绝对关键。

---

## 第 18 页｜Tuning Occupancy: Launch Bounds and the Occupancy API（调占用率的两件工具）

🎤 **讲稿**：
有时候"多开线程"还不够——尤其当每个线程本身很"重"（寄存器用得多）的时候。两件工具：
第一，**`__launch_bounds__(256, 16)`** 注解，编译期给编译器两个承诺/请求：这个 kernel 的 block 绝不超过 256 线程；请保证每个 SM 至少能驻 16 个这样的 block。编译器拿到这个信息，就会**主动压缩每线程的寄存器用量、克制内联和展开**，好让更多 warp 塞得下。有意思的细节：16×256=4096 超过了 SM 的 2048 线程上限，编译器会把请求**削到硬件上限**（8 个 block）并给出 ptxas 警告——这说明这些参数是"愿望"，硬件上限说了算。
第二，**运行时的 Occupancy API**：`cudaOccupancyMaxPotentialBlockSize()` 根据 kernel 实际的寄存器/共享内存消耗，自动算出占用率最优的 block size。一个常见的坑：它返回的 `minGridSize` 是"喂饱这块 GPU 所需的最小 grid"，**不是**覆盖你 N 个数据所需的 grid——真正的 grid 要取两者的较大值：`max(minGridSize, ceil(N/blockSize))`。如果 kernel 用了动态共享内存，还得把字节数如实传进去。
底下的权衡再强调一遍：**寄存器少 → warp 多 → 延迟藏得好；但太少 → 溢出到 local memory → 更慢**。书里的忠告：API 给的建议要**实测 ±1–2 档 block size 验证**——现代 GPU 上，略低于最大占用率的配置有时反而更快。
大白话：**你在拿"每个工人的工具箱大小"换"车间里能站多少工人"——甜点位置只能靠实验找。**

📚 **原书精读**：
> "We are essentially trading a bit of per-thread performance ... in exchange for more consistent warp throughput by keeping more warps in flight."
> 我们本质上是牺牲一点单线程性能……换取更多 warp 在飞、从而更稳定的 warp 吞吐。

---

## 第 19 页｜Debugging Correctness: NVIDIA Compute Sanitizer（正确性调试）

🎤 **讲稿**：
换个话题喘口气——不谈性能，谈正确性。一个 kernel 几千上万个线程，传统 debugger 抓不住那些偶发的内存错误和竞争条件。CUDA Toolkit 自带的 **Compute Sanitizer** 在运行时给代码"插桩"，四件套各管一摊：
- **memcheck**：越界访问、未对齐访问、显存泄漏——最常用；
- **racecheck**：共享内存上的数据竞争（写后写 WAW、读后写 WAR、写后读 RAW 三种冒险）；
- **initcheck**：读了没初始化的全局显存——典型病因是**忘了做 host 到 device 的拷贝**;
- **synccheck**：非法同步原语，比如错配的 barrier，会导致死锁或状态不一致。
用法就是命令行 `compute-sanitizer --tool <名字> ./你的程序`，可以用 `--kernel-name` 只查特定 kernel，配合 NVTX 标注缩小范围。书里的最佳实践是把它塞进 **CI**（持续集成），加 `--error-exitcode` 让出错时构建直接失败——正确性回归在合码前就被拦下。
大白话：**十万个线程的程序，"我这儿跑了没问题"毫无意义——sanitizer 是那些你永远手动复现不出来的竞争条件的安全带。**

📚 **原书精读**：
> "Since CUDA applications can spawn thousands of threads per kernel, traditional debugging may fail to catch subtle memory bugs and race conditions."
> 由于 CUDA 应用每个 kernel 能派生几千个线程，传统调试可能抓不住细微的内存 bug 和竞争条件。

---

## 第 20 页｜The Roofline Model: Which Wall Are You Hitting?（Roofline：你撞的是哪面墙）⭐重点页，建议讲 3 分钟

🎤 **讲稿**：
最后一个大概念，也是全书反复用的分析框架。看右图（Figure 6-21）。
横轴是 **arithmetic intensity（算术强度）**：**每从显存搬 1 个字节，你做了几次浮点运算**，单位 FLOPs/byte。纵轴是实际达到的性能。图上有两条"天花板"：一条**水平线**是芯片的峰值算力（Blackwell FP32 约 80 TFLOP/s）——你算得再欢也超不过它；一条**斜线**是显存带宽的上限（约 8 TB/s）——算术强度低的时候，性能 = 强度 × 带宽，被这条斜线压着。两线交点叫 **ridge point（脊点）**，Blackwell 约在 **10 FLOPs/byte**（80T ÷ 8T）。落在脊点**左边**就是 **memory-bound（访存受限）**——ALU 喂不饱，加算力没用；落在**右边**是 **compute-bound（算力受限）**。
算一遍书里的例子就全懂了：向量加法 C = A + B，每个元素要**读两个 float 进来（8 字节）、写一个回去（4 字节）**，共 12 字节的流量，只做 **1 次加法**。算术强度 = 1/12 ≈ **0.083 FLOPs/byte**——离脊点差了 **100 多倍**，在图上钉死在斜线上。结论：这个 kernel 无论怎么优化线程配置，都是访存受限——这就是为什么上一页说"占用率救不了它"。
对到大家的日常：**LLM 的 prefill（读题）阶段偏 compute-bound，decode（逐 token 生成）阶段偏 memory-bound**——因为 decode 每生成一个 token 都要把权重从 HBM 过一遍。第 15–18 章讲推理时会大量用到这个视角。
大白话：**roofline 在你动手优化前先问一个问题：这个 kernel 是"饿计算"还是"饿数据"？答错方向，白干。**

📖 **页面英文对照**：
- ridge point = 脊点；compute roof = 计算屋顶（水平线）；memory roof = 内存屋顶（斜线）。
- "hopelessly memory-bound" → 无可救药地访存受限。

📚 **原书精读**：
> "Where these lines intersect is called the ridge point. This corresponds to the 'arithmetic intensity' threshold at which a kernel transitions from being memory bound (left of the ridge) to compute bound (right of the ridge)."
> 两条线的交点叫脊点，对应一个算术强度阈值：kernel 在脊点左侧是访存受限，右侧是算力受限。

---

## 第 21 页｜Moving Right on the Roofline: Lower Precision Pays Twice（在 roofline 上向右移动：低精度一石二鸟）

🎤 **讲稿**：
知道了自己 memory-bound，怎么办？方向只有一个：**提高算术强度，往图的右边挪**。要么每个字节多干活（片上复用、算子融合），要么——更直接——**把字节本身变小**。
关键账目：GPU 访存的基本单位是 **128 字节一个事务**。这一个事务能装 **32 个 FP32，或 64 个 FP16，或 128 个 FP8，或 256 个 FP4**。也就是说，从 FP32 换到 FP16，同样的带宽下数据量翻倍、算术强度**立刻翻倍**——kernel 在 roofline 上直接右移一格。Blackwell 原生支持 FP8 和 FP4 的 Tensor Core，把这条路修到了 4 位。
更妙的是 Blackwell 的**硬件解压（hardware decompression）**：权重可以以压缩形式存在 HBM 里（甚至比 FP4 更狠的 2 位方案），读取时硬件**现场解压**、再转成 FP16/FP32 做高精度累加——相当于**变相扩大了可用显存带宽**。这正是 Blackwell 跑 memory-bound 的 **token 生成（decode）** 特别强的架构原因。
下面蓝色框里的话是这一页的灵魂，值得慢慢念：**精度既是计算优化，更是"带宽优化"——字节减半 ⇒ 算术强度翻倍 ⇒ 向计算屋顶靠拢。**

📚 **原书精读**：
> "A single 128-byte memory transaction can carry 32 FP32, 64 FP16, 128 FP8, or 256 FP4 values."
> 一个 128 字节的访存事务可以携带 32 个 FP32、64 个 FP16、128 个 FP8 或 256 个 FP4 值。
> "models can be stored compressed in HBM ... and the hardware can decompress the weights on the fly. This effectively increases the usable memory bandwidth."
> 模型可以压缩存放在 HBM 里……硬件在读取时实时解压。这实质上提高了可用的显存带宽。

---

## 第 22 页｜Profiling Workflow: nsys for When, ncu for Why（性能分析工作流）

🎤 **讲稿**：
最后是方法论。两个工具，分工记这一句就行：**nsys 管"什么时候"，ncu 管"为什么"**。
- **Nsight Systems（nsys）**：整个应用的时间线——GPU 什么时候在空转、CPU 和 GPU 有没有重叠、PCIe/NVLink 传输在哪。原书的说法：**整台机器的秒表**。
- **Nsight Compute（ncu）**：钻进单个 kernel 的计数器——占用率多少、warp 停在什么原因上（stall reasons）、缓存命中率、DRAM 利用率。原书的说法：**对准一个 kernel 的显微镜**。
两个必须搭配用：只跑 nsys，看不到 kernel 内部为什么慢；只跑 ncu，不知道 kernel 是不是根本"没被喂上数据"。再加上 **NVTX**——在代码里给可疑区域打标签，时间线上直接显形。
两个实战鉴别式：① memory-bound 的特征签名 = ncu 里 **DRAM 利用率很高 + ALU 利用率很低**，说明 warp 大部分时间停在访存上；② 期望传输和计算重叠（右图 Figure 6-22 下半部分）却看到它们串行执行？两大常见病因：**忘了用 pinned memory（页锁定内存）**——没有它 cudaMemcpyAsync 根本无法和 kernel 重叠;或者默认流的隐式同步在作怪。
收尾原则：**每改一处，测一次**——profiler 会告诉你这次优化是真减少了 stall，还是只是心理安慰。

📚 **原书精读**：
> "Without using pinned (page-locked) memory, the cudaMemcpyAsync transfer cannot overlap with kernel execution. This is a common performance issue."
> 不用 pinned（页锁定）内存，cudaMemcpyAsync 就无法与 kernel 执行重叠。这是一个常见的性能问题。
> "Together they give you both the why (which stalls and which resources) and the when (how those stalls fit into your application's overall execution)."
> 两者合起来，你既有了"为什么"（哪些停顿、哪些资源），也有了"什么时候"（这些停顿嵌在整个应用执行的哪里）。

---

## 第 23 页｜Key Takeaways（要点回顾）

🎤 **讲稿**：
（这一页照左右两栏念要点即可，每条一句话）
左栏：**SIMT**——32 线程的 warp 齐步走，多 warp 在飞才能藏延迟；**线程层级**——thread → block（≤1024）→ grid，block 互相独立所以可移植地扩展；**block 尺寸**——32 的倍数、从 256 起步，记住每 SM 的四个上限（64 warp / 32 block / 228 KB 共享内存 / 每线程 255 寄存器）；**异步分配**——cudaMallocAsync + stream + 内存池，PyTorch 的分配器已经替你做了。
右栏：**内存梯子**——寄存器→共享/L1→L2→HBM，上层复用、下层合并访存；**统一内存**——方便，但要 prefetch + advise 才不会被缺页偷袭；**roofline**——FLOPs/byte 决定你是饿数据还是饿计算，降精度让你右移；最后一条最重要——**occupancy 是必要条件而非充分条件**：高占用率治不了带宽瓶颈，先 profile 再动手。

---

## 第 24 页｜Conclusion and What's Next（结论与下一步）

🎤 **讲稿**：
整章浓缩成一句话（蓝框）：**让 GPU 忙起来（occupancy）、让数据离计算近一点（内存层级）、让 roofline 告诉你下一仗往哪打。**
一个反直觉的收尾提醒，来自原书结论：**占用率最大化并不总是最优解**。如果每个线程内部有足够的指令级并行（ILP，instruction-level parallelism），中等甚至偏低的占用率也能跑满吞吐；有时候**故意少开线程、让每个线程多拿寄存器**反而更快。判断标准只有一个：benchmark。
预告一下后续：第 7、8 章讲访存模式和 warp 效率的深度调优，第 9 章讲算术强度，第 10 章讲 TMA 和 warp specialization。我负责的另一章——**第 12 章**——和今天正好衔接：今天解决"单个 kernel 怎么快"，第 12 章解决"kernel 之间怎么编排，让 GPU 永远不用等 CPU"。
谢谢大家，欢迎提问。

📚 **原书精读**（结论段的关键转折）：
> "However, maximizing occupancy does not guarantee best performance in every case. GPUs can often achieve very high throughput at moderate or even low occupancy if threads have sufficient instruction-level parallelism (ILP)."
> 然而，占用率最大化并不能保证在所有情况下性能最优。只要线程有足够的指令级并行，GPU 常常能在中等甚至较低的占用率下达到很高的吞吐。

---

## 第 25 页｜References（参考文献）

🎤 **讲稿**：
参考资料放在这页：原书第六章；NVIDIA 的 Blackwell 调优指南和 CUDA C++ 编程指南（本章所有硬件上限的权威出处）；roofline 模型的原始论文——Williams、Waterman、Patterson 2009 年发在《Communications of the ACM》上的经典；以及 Compute Sanitizer 和 Nsight 两件套的官方文档。表格里的性能数字是书里的示意值，真实的分架构 benchmark 在书的 GitHub 仓库。

---

# 附录 A｜幻灯片没展开、但书里有的内容（防提问）

1. **CUDA 的前后向兼容模型（书 202–203 页）**：编译产物分两种——**SASS**（特定架构的机器码）和 **PTX**（虚拟指令集，可以在新架构上 JIT 即时编译）。只带 sm_90 的 SASS 不带 PTX 的二进制**上不了新架构**；最佳实践是打 **fatbin**（胖二进制）：通用 PTX + 需要的架构专用 cubin 都塞进去。可以设环境变量 `CUDA_FORCE_PTX_JIT=1` 强制走 PTX JIT 来验证兼容性——如果二进制里没有 PTX，kernel 启动会直接失败。还有一类 `sm_100f` 带 f 的"家族"目标，只在同特性家族内可移植。
2. **Thread block cluster / DSMEM（书 196–197 页）**：cluster 内不同 block 的线程可以互访彼此的共享内存、用簇级 barrier。DSMEM 靠片上高速互连把多个 SM 的共享内存 bank 连成一个池子，读写和原子操作都不占全局显存带宽。第 10 章细讲。
3. **2D kernel 完整代码（书 209 页）**：`my2DKernel` 用 `int x = blockIdx.x*blockDim.x+threadIdx.x; int y = ...` 算二维坐标，`idx = y*width + x` 摊平成一维下标，边界检查 `if (x<width && y<height)`。
4. **内存池调优（书 211 页）**：`cudaMemPoolAttrReleaseThreshold` 提示池子保留多少内存不还给系统；`cudaMemPoolTrimTo` 主动归还。在"总显存占用"和"碎片化"之间找平衡。
5. **点熟 nsys/ncu 命令行（书 226 页）**：`nsys profile --stats=true -t cuda,nvtx -o report ./app`；`ncu --section SpeedOfLight --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./app`。ncu 里 achieved occupancy 的指标名就是这个 `sm__warps_active...`。
6. **coherency 的层级（书 217 页）**：内存一致性的"生效点"（point of coherency）按需要发生在 thread / thread block / cluster / device / system 不同层级——线程间通信的层级越广，代价越大。

# 附录 B｜术语总表（英文 → 中文 → 一句话）

| 英文 | 中文 | 一句话 |
|---|---|---|
| host / device | 主机 / 设备 | CPU 侧 / GPU 侧 |
| kernel | 核函数 | 跑在 GPU 上的函数，`__global__` 标注 |
| SM | 流式多处理器 | GPU 的"车间"，Blackwell 有一百多个 |
| thread / thread block (CTA) / grid | 线程 / 线程块 / 网格 | 工人 / 班组（≤1024 人）/ 全部班组 |
| warp | 线程束 | 32 线程的最小调度单位，齐步走 |
| SIMT | 单指令多线程 | 一条指令驱动 32 个线程 |
| warp scheduler | warp 调度器 | 每 SM 四个，各自每拍发射一个 warp |
| dual-issue | 双发射 | 同 warp 同拍发 1 算术 + 1 访存 |
| SFU | 特殊功能单元 | sin/cos/sqrt 专用管线 |
| occupancy | 占用率 | 活跃 warp ÷ 上限 64 |
| achieved occupancy | 实际占用率 | profiler 实测值，区别于理论值 |
| latency hiding | 延迟隐藏 | 等内存时切换别的 warp |
| warp divergence | warp 分叉 | 同 warp 内走不同分支 → 串行化 |
| lane / mask | 车道 / 屏蔽 | warp 里的一个线程位 / 分叉时关掉不走这条路的车道 |
| coalesced access | 合并访存 | 128 字节对齐的连续访问，一次事务搞定 |
| register spilling | 寄存器溢出 | 寄存器不够、数据被挤到慢速 local memory |
| local memory | 局部内存 | 名字有欺骗性：物理上在 DRAM，很慢 |
| shared memory (SMEM) | 共享内存 | block 内共享的片上 SRAM |
| bank conflict | 存储体冲突 | 多线程撞到共享内存同一 bank，串行化 |
| constant memory | 常量内存 | 64 KB 只读区 + 8 KB 缓存，同地址读可广播 |
| TMEM | 张量内存 | Blackwell 每 SM 256 KB，Tensor Core 的累加器 |
| TMA | 张量内存加速器 | 按描述符自动搬运 HBM↔SMEM↔TMEM |
| UMMA / tcgen05 | 统一矩阵乘累加 / 第五代 TC 指令 | Blackwell Tensor Core 的指令家族 |
| HBM3e | 高带宽显存 | B200：180 GB、约 8 TB/s |
| Unified Memory / managed memory | 统一内存 / 托管内存 | CPU+GPU 一个地址空间，页自动迁移 |
| page fault / migration | 缺页 / 迁移 | GPU 摸到不在本地的页 → 停下来等搬运 |
| pinned memory | 页锁定内存 | 不会被 OS 换页的主机内存，异步拷贝的前提 |
| stream | 流 | GPU 上的操作队列，"传送带" |
| memory pool | 内存池 | 复用已释放显存，避免 OS 调用 |
| `__launch_bounds__` | 启动边界注解 | 编译期承诺 block 上限、请求驻留数 |
| arithmetic intensity | 算术强度 | FLOPs ÷ 搬运字节数 |
| roofline / ridge point | 屋顶线 / 脊点 | 两条性能天花板 / 访存与算力受限的分界（Blackwell ≈ 10 FLOPs/B）|
| memory-bound / compute-bound | 访存受限 / 算力受限 | 饿数据 / 饿计算 |
| ILP | 指令级并行 | 单线程内多条指令可并行，低占用率的补偿手段 |
| NVTX | NVIDIA 工具扩展 | 给代码打时间线标签 |
| Nsight Systems / Compute | — | 整机秒表 / 单 kernel 显微镜 |
| Compute Sanitizer | 计算消毒器 | memcheck/racecheck/initcheck/synccheck 四件套 |
| fatbin / PTX / SASS | 胖二进制 / 虚拟汇编 / 机器码 | 兼容性三件套：PTX 保前向兼容 |
