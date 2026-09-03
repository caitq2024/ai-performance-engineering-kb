# 第六章 Part IV 讲稿（预览稿，对应 chapter06-part4-draft.pdf，共 11 页）

> **说明**：Part IV = 原书 Maintaining High Occupancy and GPU Utilization（p249）+ Tuning Occupancy with Launch Bounds（p257）两个小节。这是全章的**兑现时刻**：Part I 讲的 occupancy 理论，在一个 22 倍加速的案例里落地。页码为预览稿编号。
> 每节结构：🎤 口播稿（覆盖页面全部要点）／📖 页面对照（幻灯片文字的中文版）／大白话／❓ 预备问答。

---

## 第 1 页｜标题页

🎤 进入 Part IV，全章的兑现时刻。转场词（对应第 2 页顶部灰字）："内存层级告诉我们 warp 为什么会 stall。现在回到 occupancy：如果一个 warp 在等数据，我们究竟需要多少其他 warp 才能填满这些空档？"这一部分只做一件事：同一个运算、两种写法、profiler 当裁判——**22 倍**。

---

## 第 2 页｜Occupancy Ground Rules（占用率基本法则）⭐

🎤 蓝框是书里"CUDA 性能最基本的法则"：**"Launch enough parallel work to fully occupy the GPU."——启动足够多的并行工作，把 GPU 填满。**
下面两条规则务必分清，它们是这一部分的纲：
**规则一——occupancy 低且性能差**：第一味药是**加并行度**（更多 block/线程），加到"有足够多的 ready warp 能把延迟藏住"为止——**性能不再上涨（plateau）就停**，不要以某个固定百分比为目标（第 8 页 38.7% 拿到 22 倍就是证据）。
**规则二——occupancy 已经中高但还是慢**：如果 kernel 是 **memory-bound**，硬推到 100% **没用**——你只需要"刚好够藏延迟"的 warp 数，超过之后瓶颈在带宽，不在人手。
预告：接下来 5 页是一个完整案例——同一个操作（C = A + B，一百万元素）、两种实现、两个 profiler 的裁决。

📖 页面对照：
- 蓝框 "Launch enough parallel work to fully occupy the GPU." → 启动足够多的并行工作填满 GPU（书中基本法则）。
- Rule 1 "increase parallelism until there are enough ready warps to hide latency --- stop when performance plateaus" → 加并行度直到就绪 warp 够藏延迟；性能不涨就停。
- Rule 2 "if the kernel is memory-bound, pushing to 100% won't help" → memory-bound 时推满无益。

大白话：**规则一是让车间站满小组；规则二提醒你：车间站满了，卡在仓库运货卡车上照样停工。**

❓ 可能被问："occupancy 到底多少算够？"→ 没有普适数字——"够"的定义是**把停顿藏住、性能进入平台期**。案例里 38.7% 就够了；算术密集的 kernel 可能 25% 就饱和；实测为准。

---

## 第 3 页｜Case Study, Take 1: addSequential（案例上：串行版）⭐

🎤 案例第一回合，先看一个"反面教材"kernel。左边代码：`addSequential`——**一个线程做全部一百万次加法**。注意两处：函数体里 `if (blockIdx.x == 0 && threadIdx.x == 0)` 把活儿全锁给 0 号线程，一个 for 循环从头加到尾；启动配置是 **`<<<1, 1>>>`**——1 个 block、1 个线程。
右边三个要点：① 一个线程 = 一个 warp = 一个 SM 在干活——**其余全体围观**：B200 上一百多个 SM，一百多减一个全在闲置；② 书里的评语（意译）：GPU 庞大的资源基本闲置，占用率极差、性能自然极低；③ 最致命的机制性问题：**没有其他 warp 可切换**——每次访存等待都变成**纯粹的空转**，Part I 讲的延迟隐藏完全失效，因为调度员手里只有一支队伍。

📖 页面对照：
- 代码注释 "Single thread does ALL N additions" → 单线程做全部 N 次加法；"this kernel assumes <<<1,1>>>" → 该 kernel 假定 1 block × 1 线程启动。
- "One thread, one warp, one SM --- everything else watches." → 一线程一 warp 一 SM，其余围观。
- "No other warps to switch to → memory waits become pure idle time --- zero latency hiding." → 无 warp 可切换，等待变纯空转，延迟隐藏为零。

大白话：**你租下了整座工厂，然后只雇了一名工人。**

❓ 可能被问："谁会真写 `<<<1,1>>>`？"→ 显式没人写，但**隐式的串行段很常见**：没并行化的预处理/后处理、按样本循环调 kernel、自定义算子里"先算个全局统计量"的串行小尾巴。这个极端例子是用来给 profiler 指标定标尺的。

---

## 第 4 页｜The Same Trap in PyTorch（同一个坑的 PyTorch 版）⭐

🎤 别以为用 PyTorch 就免疫了——同一个坑换了件衣服。左边代码：Python 里 `for i in range(N): C[i] = A[i] + B[i]`，对一百万个元素逐个下标赋值。
三个要点：① 这不是"一个 kernel 跑一百万次循环"，而是**串行发射一百万个迷你 kernel**——每次 `C[i] = A[i] + B[i]` 都是一次独立的 GPU 操作，书里的说法：GPU 沦为"一个标量的、毫无并行的处理器"；② 占用率和上一页一样惨，还要**外加每次 launch 的 CPU 开销 × 1,000,000**——CPU 提交开销通常几微秒，光发射就要几秒；③ 书的建议：几乎总能找到 **PyTorch 原生的向量化写法**（包括 torch.compile 自动生成的）——下一页就是。
这一页对我们组是最实用的一页：**Python 循环套 GPU 逐元素操作，是生产代码里最常见的性能事故**。

📖 页面对照：
- 代码注释 "Naive, sequential GPU ops - DO NOT DO THIS" → 朴素串行 GPU 操作——不要这样写。
- "a Python loop around GPU ops serializes N tiny kernel launches" → Python 循环把 N 个迷你 kernel 串行化。
- "there is almost always an optimized PyTorch-native (vectorized) implementation" → 几乎总有 PyTorch 原生向量化实现。

大白话：**最常见的 GPU 性能 bug：一不小心把 GPU 当成了一颗非常昂贵的单核 CPU。**

❓ 可能被问："怎么发现自己踩了这个坑？"→ nsys 时间线一眼可见：**密密麻麻的微型 kernel**，每个之间还有 CPU 提交间隙，GPU 利用率个位数。修法：改写成张量化操作，或交给 `torch.compile`。

---

## 第 5 页｜Case Study, Take 2: addParallel — and C = A + B（案例下：并行版）⭐

🎤 正确写法两种形态。左边 CUDA 版 `addParallel`：**一个线程管一个元素**——就是 Part II 学的标准骨架（算 idx、边界检查、干活），启动 `<<<(N+255)/256, 256>>>`。注意参数上的新面孔 **`__restrict__`**：程序员向编译器承诺"这几个指针互不重叠（无别名）"，编译器就敢放心地缓存和重排访存——解开优化的手脚。
左下小字别漏：书里的完整版本还用了**锁页主机内存、非阻塞流、cudaMallocAsync/cudaMemcpyAsync**——正是 Part II 养成的全部习惯，这里全用上了。完整版**下一页顺手带过**——这页先聚焦"一个线程 vs 一百万个线程"这个对比本身。
右边三个要点：① 约 3,907 个 block × 256 线程——**一百万个线程**铺满数组（图 6-19），Part I 实例页的配置原样重现；② PyTorch 等价写法就一行：**`C = A + B`**——单个向量化 kernel，同时调动海量线程；③ 对比上一页：同样的数学，写法一变，GPU 从"单核 CPU"变回吞吐量机器。

📖 页面对照：
- "One thread per element" → 一线程一元素。
- "__restrict__: promises no pointer aliasing --- frees the compiler to optimize" → 承诺指针无别名，解放编译器。
- "PyTorch equivalent is one line: C = A + B --- a single vectorized kernel engaging many threads concurrently" → PyTorch 等价一行，单个向量化 kernel 驱动海量线程。

大白话：**还是那座工厂，现在每个工位都有人——PyTorch 版等于直接"找劳务派遣公司包了"。**

❓ 可能被问："`__restrict__` 到底干了什么？"→ C99 的 `restrict` 在 CUDA 里的写法。没有它，编译器必须假设 A、B、C 可能指向同一块内存（写 C 可能改掉 A），于是每次都老老实实重新加载；有了它，编译器可以把数据留在寄存器、合并访存、重排指令。逐元素 kernel 的惯用签名。

---

## 第 6 页｜addParallel, Full Version: The Part II Habits Assembled（完整版：Part II 习惯大集合）

🎤 **顺手带过，30–40 秒，不逐行念**。上一页的 kernel 一个字没变，这页只是给主机侧换上正式工装——**每一行都是 Part II 教过的**：建非阻塞流；`cudaMallocHost` 锁页分配（六步流程第 1 步）；`cudaMallocAsync` 从池里取显存（第 8 页）；然后是唯一的新面孔 **`cudaMemcpyAsync`**——异步拷贝，**排进流 s 里**，它能与计算重叠的前提正是主机缓冲区是锁页的（Part II 第 3 页的伏笔）；kernel 也发进同一条流；`cudaStreamSynchronize(s)` **只等这一条流**，不做全局同步；最后 `cudaFreeAsync` 异步归还。
一句话：同一个 kernel，生产级的管线——Part II 的习惯在这里全部组装完毕。

📖 页面对照：
- "cudaMemcpyAsync ... can overlap with compute only because the host buffer is pinned" → 异步拷贝能与计算重叠，前提是主机缓冲区锁页。
- "cudaStreamSynchronize(s): wait for this stream only --- no global cudaDeviceSynchronize" → 只等本流，不全局同步。

大白话：**同一个 kernel，换上正式工装——Part II 的习惯全套上身。**
## 第 7 页｜Measuring It: nsys and ncu（怎么量：两件裁判工具）

🎤 空口无凭，两件工具当裁判，各管一摊：
**nsys（Nsight Systems）**——整机秒表。命令：`nsys profile --stats=true -t cuda,nvtx -o report <程序>`。它回答"**时间花到哪儿去了**"：GPU 是没喂饱（starved）还是被堵住（blocked）？kernel 之间有没有空档？
**ncu（Nsight Compute）**——单 kernel 显微镜。命令里那个超长的指标 `sm__warps_active.avg.pct_of_peak_sustained_active` **就是 achieved occupancy 本尊**——实测占用率在工具里的学名。它回答"**这个 kernel 为什么慢**"：占用率？停顿？缓存未命中？
为什么两个都要跑：只跑 nsys 看不到 kernel **内部**的低效；只跑 ncu 看不出 kernel 之间数据**喂得够不够快**。先 nsys 找到时间大头，再 ncu 解剖那个大头——标准工作流。

📖 页面对照：
- "nsys answers 'where does time go --- is the GPU starved or blocked?'" → nsys 回答时间去哪了。
- "ncu answers 'why is this kernel slow --- occupancy? stalls? cache misses?'" → ncu 回答这个 kernel 为什么慢。
- "That long sm__warps_active... metric is achieved occupancy." → 那个长指标就是实测占用率。

大白话：**nsys 是给整台机器计时的秒表；ncu 是对准单个 kernel 的显微镜。**

❓ 可能被问："profiler 本身拖不拖慢程序？"→ nsys 开销小（时间线采样）；ncu 开销大——它会**重放（replay）** kernel 多次来采不同计数器，所以用 `--kernel-name` 只对目标 kernel 下手，别全量跑。

---

## 第 8 页｜Parallelism Cuts Runtime 22× at Only 38.7% Occupancy（裁决：22 倍）⭐

🎤 裁决书（表 6-6，示意值，真实数据在书的 GitHub 仓库）。四行逐行读：
**Kernel 耗时**：48.21 ms → **2.17 ms**，**22 倍**——本章标题"Maximizing Occupancy"的兑现。
**GPU 利用率**：1.5% → 95%。
**实测占用率**：1.3% → **38.7%**。
**Warp 执行效率**：3.1% → 100%——3.1% 这个数很有讲头：**3.1% ≈ 1/32**，串行版每个 warp 只有一条通道在干活，正是"一个人的赛艇队"的数字化呈现。
右图（图 6-20）是加速的本质：串行版时间线是"操作—空档—操作—空档"；并行版**空档被其他 warp 的工作填满**——Part I 延迟隐藏的可视化终章。
两个洞察：① **38.7% 就买到了 22 倍——根本不需要 100%**（回收第 2 页规则一的"不设固定目标"）；② 表里四个指标——utilization、SM active、occupancy、warp efficiency——**相关但不同维度**：95%、38.7%、100% 能同时出现，正说明它们量的不是一件事。

📖 页面对照：
- 表 6-6 四行（时间/利用率/占用率/warp 效率）及"illustrative --- real benchmarks in the book's repo"→ 示意值，真实基准见配套仓库。
- "Warp efficiency 3.1% ≈ 1/32: one active lane per warp." → 每 warp 仅一条活跃通道。
- "GPU utilization, SM active %, achieved occupancy, and warp efficiency are related but different metrics" → 四个相关但不同的指标。

大白话：**38.7% 的上座率就赚回 22 倍——先把人招够，别急着追满座。**

❓ 可能被问："为什么 38.7% 就够了？"→ 回到 occupancy 的定义（Part I 第 16 页）：它是"调度员手里有多少候选"。38.7% × 64 ≈ 25 个常驻 warp/SM，对这个访存模式已经足够在每次停顿时找到就绪者——停顿被藏住后，再加 warp 就没增量了。

---

## 第 9 页｜A Busy GPU Can Still Be Waiting on Memory (LLM Decode)（忙碌的 GPU 仍可能在等内存）⭐

🎤 顶部灰字是全场最重要的转场："并行版达到 95% utilization、只有 38.7% occupancy、已经快了 22 倍——所以我们不需要 100%。但它到 GPU 算力峰值了吗？**仍然没有，因为 vector add 主要在搬数据。**"
三个要点：① 并行度拉满之后，下一个抓手是单 warp 效率（ILP 等，第 8 章）——但**哪怕 occupancy 100%**，只要 kernel 是 **memory-bound**（受制于数据搬运而非计算），性能照样上不去；② 书里的经典例子：**LLM 的 decode 阶段**——每生成一个 token，都要把**模型权重**从 HBM 整个流进片上；几千亿参数 × 约 1 字节 ≈ **每个 token 几百 GB** 的搬运量——多少线程都救不了带宽；③ 蓝框是让情况雪上加霜的趋势：**GPU 算力增速超过内存带宽**——HBM3e 约 8 TB/s，但算力和模型规模涨得更快，所以"优化数据搬运"在现代 AI 负载里绝对关键。
这页把听众从"occupancy 万能"的错觉里拽出来，为 Part V 的 roofline 铺路。

📖 页面对照：
- "even at 100% occupancy, performance suffers if the kernel is memory bound" → 100% 占用率也救不了 memory-bound。
- "the decode phase of an LLM --- every generated token streams model weights from HBM" → decode 每个 token 都要从 HBM 流一遍权重。
- 蓝框 "GPU FLOPS are outpacing memory bandwidth... optimizing memory movement is absolutely critical" → 算力增速超带宽；优化搬运绝对关键。

大白话：**当整个活儿就是"从仓库搬箱子"，前台多雇几个职员没有用——这就是 Part V 屋顶线要形式化的事。**

❓ 可能被问："为什么 decode 每个 token 都要读全部权重？"→ 自回归生成一次只算一个 token，矩阵乘退化成"向量 × 矩阵"，每个权重只被用一次（复用率 = 1）——读进来做一次乘加就扔。batch 加大才能摊薄（多个序列共享同一遍权重读取），这正是第 15–18 章 continuous batching 的动机。

---

## 第 10 页｜__launch_bounds__: Compile-Time Occupancy Control（编译期的占用率控制）

🎤 案例讲完，最后两页是两件调优工具。第一件在**编译期**：`__global__ __launch_bounds__(256, 16)`——两个参数是你对编译器的**承诺和请求**：承诺"这个 kernel 启动时每 block 不超过 256 线程"；请求"每 SM 保持至少 16 个 block 常驻"。
左下的算术：16 × 256 = 4,096 > 2,048 的硬件上限 → 编译器**压回** 8 个 block，还会给出 `ptxas warning`——所以参数要按 Part I 的上限表算好。
右边三个要点：① 拿到承诺后编译器做什么：**限制每线程寄存器数**（可能还收紧循环展开和内联），以避免溢出、让每 SM 装下更多 warp；② 这笔交易的本质：**牺牲一点单线程性能，换更稳定的 warp 吞吐**（更多 warp 在飞）；③ 危险区：硬塞太多线程 → 寄存器不够 → **溢出到 local memory**（Part III 第 3 页的悬崖）——比不调还慢。

📖 页面对照：
- 代码注释 "promise: never launched with > 256 thr/block; request: keep >= 16 blocks resident per SM" → 承诺与请求。
- "compiler caps per-thread registers... to avoid spills and fit more warps per SM" → 编译器限制每线程寄存器，防溢出、多驻 warp。
- "Danger zone: forcing too many threads → spilling to local memory --- slower than what you started with" → 硬塞过头反而更慢。

大白话：**你在拿"每个工人工具箱的大小"换"车间里工人的数量"——最佳配比只能靠实测。**

❓ 可能被问："它和下一页的 Occupancy API 什么关系？"→ 一个管**编译期**（影响编译器给每线程分多少寄存器），一个管**运行期**（据实际资源用量算最优 block 尺寸）。常配合用：`__launch_bounds__` 定资源上限，API 在上限内选启动参数。

---

## 第 11 页｜The Occupancy API: Runtime Autotuning（运行期自动调参）

🎤 第二件工具在**运行期**。核心调用：`cudaOccupancyMaxPotentialBlockSize(&minGridSize, &bestBlockSize, kernel, dynSmemBytes, 0)`——它读取这个 kernel **实际的**寄存器和共享内存用量，算出**占用率最大化的 block 尺寸**。不用自己拿上限表算，硬件换代也自动适配。
两个坑（书里特别点名）：① **`minGridSize` 不是"覆盖 N 的网格"**——它是"占满占用率所需的最小网格"；正确用法是代码里那行 `gridSize = max(minGridSize, (N + bestBlockSize - 1) / bestBlockSize)`——既要覆盖数据，也别低于占满机器的下限；② 如果 kernel 用了 `extern __shared__`，**动态共享内存字节数要如实传**——传 0 会把占用率算飘。
最后一条书中提示，呼应全章的"实测为准"：**在 API 给的最优值 ±1–2 档再各测一次**——寄存器压力和 L2 行为有时会让"略低于最大占用率"的配置反而更快。

📖 页面对照：
- "Computes the block size that maximizes occupancy given the kernel's actual register/shared-memory usage" → 按实际资源用量算出占用率最大的 block 尺寸。
- "minGridSize saturates occupancy --- it is not the grid that covers N (take the max)" → minGridSize 是占满占用率的网格，不是覆盖 N 的网格，要取 max。
- "validate with ±1--2 candidate block sizes --- sub-maximal occupancy can be faster" → 最优值附近再实测，略低占用率可能更快。

大白话：**让机器自己报一个"最合适的班组人数"——但报完还是要试跑两组对照，机器也会看走眼。**

❓ 可能被问："那第 6 页配方里的 256 还要不要？"→ 要——256 是**没有 profiler 时的起点**；Occupancy API 是**部署前的精调**。顺序：256 起步 → 能跑通出结果 → 用 API + ±1–2 档实测定终值。

---

