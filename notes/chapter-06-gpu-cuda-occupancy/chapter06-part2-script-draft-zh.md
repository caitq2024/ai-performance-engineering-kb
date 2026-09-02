# 第六章 Part II 讲稿（预览稿，对应 chapter06-part2-draft.pdf，共 9 页）

> **说明**：Part II = 原书 CUDA Programming Refresher 一节（p231–240）：kernel 解剖（设备侧/主机侧）→ 为什么传 N → 边界检查与懒惰报错 → launch 参数配方 → 2D/3D 输入 → 异步分配与内存池。页码为预览稿编号；第 6 页（256 配方）已按商定改薄为"代码兑现"页，理由部分留在 Part I。
> 每节结构：🎤 口播稿（覆盖页面全部要点）／📖 页面对照（幻灯片文字的中文版）／大白话／❓ 预备问答。

---

## 第 1 页｜标题页

🎤 （中场休整后）我们进入 Part II：CUDA Programming Refresher。到这里，书的 Understanding GPU Architecture 一节讲完了。Part I 回答了"硬件怎样执行并行工作"；这一部分回答"程序怎样描述并行工作"——刚才所有的"为什么"，现在一条条落成代码。这部分代码都很短，我们逐行过。

---

## 第 2 页｜Anatomy of a CUDA Kernel: Device Side（kernel 解剖：设备侧）⭐

🎤 这是全书第一个完整的 kernel，五行，逐行看：
第 1 行 **`__global__ void myKernel(float* input, int N)`**——`__global__` 修饰符的意思是：这个函数**在 GPU（设备）上运行，但从 CPU（主机）上调用**。参数就是一个裸指针加一个长度。
第 2 行 **`int idx = blockIdx.x * blockDim.x + threadIdx.x;`**——全页最重要的一行。三个内置变量：`blockIdx`（我在第几个 block）、`blockDim`（每个 block 多大）、`threadIdx`（我在 block 内排第几）。三者一乘一加，每个线程算出一个**全局唯一的编号**——这就是它"领到的那一格数据"。
第 3–4 行 **`if (idx < N) { input[idx] *= 2.0f; }`**——边界检查（第 5 页专门讲为什么），然后就地把自己那个元素翻倍。
右侧三个要点：① kernel 是站在**单个线程**的视角写的，`idx` 划出属于这个线程的那一小片数据；② CUDA 把它编译成设备代码，由**成千上万乃至上百万**个轻量线程并行执行——你写一份，它跑一百万份；③ 主机侧的启动语法 **`<<<blocksPerGrid, threadsPerBlock>>>`**——每一次 kernel 调用都绕不开的两个核心参数（Part I 实例页已经见过了）。

📖 页面对照：
- `__global__`: runs on the device, callable from the host → 在设备上运行、从主机调用。
- built-ins: blockIdx（我在哪个 block）、blockDim（block 多大）、threadIdx（我在 block 里排第几）。
- "The kernel is written from the perspective of one thread; idx carves out its slice of the data." → kernel 以单个线程的视角编写；idx 切出属于它的数据分片。
- "CUDA compiles it into device code executed by thousands or millions of lightweight threads in parallel." → CUDA 把它编译成由成千上万轻量线程并行执行的设备代码。

大白话：**你只写一名工人的岗位说明书；CUDA 复印一百万份，每份填上不同的行号。**

❓ 可能被问："这是什么语言？CUDA 是一门新语言吗？"→ 是 **CUDA C++**：主体就是标准 C++，扩展只有几样——`__global__` 这类修饰符、`<<< >>>` 启动语法、`blockIdx` 等内置变量，外加 `cuda*` 运行时 API（这是函数库不是语法）。编译器 nvcc 把一份 .cu 劈成两半：主机代码走普通 C++ 编译器，kernel 编成 PTX/SASS（呼应 Part I 兼容性页）。会 C++ 就会 90%。

---

## 第 3 页｜Host Side: The Six-Step Data Flow（主机侧：六步数据流）

🎤 kernel 写好了，主机侧要干六件事把它伺候起来。左边代码从上往下：
**第 1 步：分配**。`cudaMallocHost(&h_input, ...)` 在主机上分配**锁页（pinned）内存**——注意不是普通 malloc；然后初始化数据；`cudaMalloc(&d_input, ...)` 在 GPU 显存上分配同样大小。命名习惯全书通用：**`h_` 前缀 = host 指针，`d_` 前缀 = device 指针**——两个地址空间，千万别拿错。
**第 2 步：拷过去**。`cudaMemcpy(d_input, h_input, ..., cudaMemcpyHostToDevice)`——H2D。
**第 3 步：启动**。算好 `threadsPerBlock = 256`、`blocksPerGrid = (N + 255) / 256`（一百万元素就是 3,907 个 block），然后 `myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N)`。
**第 4 步：等**。`cudaDeviceSynchronize()`——kernel 是异步启动的，主机要显式等 GPU 干完。
**第 5 步：拷回来**。`cudaMemcpy(..., cudaMemcpyDeviceToHost)`——D2H。
**第 6 步：清理**。`cudaFree` + `cudaFreeHost` 配对释放。
右侧两个要点：① 为什么用 `cudaMallocHost` 而不是 malloc——**锁页内存是后面异步拷贝真正实现"传输与计算重叠"的先决条件**（第 8 页和 Part V 会兑现这个伏笔）；② 书里自己说明：这个版本**还没优化**——它是一份简单而完整的**模板**，全书后面的章节都在改进它。

📖 页面对照：
- naming convention: h_ = host pointer, d_ = device pointer → 命名惯例，全书通用。
- "cudaMallocHost = pinned (page-locked) memory --- prerequisite for async copies to truly overlap later" → 锁页内存：日后异步拷贝要真正重叠的先决条件。
- book note: "not fully optimized yet --- this is the simple, complete template the rest of the book improves on" → 尚未充分优化——这是全书后面不断改进的简单完整模板。

❓ 可能被问："锁页是什么意思？"→ 操作系统承诺这块内存**不会被换页/挪动**，物理地址固定，GPU 的 DMA 引擎才能绕过 CPU 直接搬运——这是异步拷贝的硬件前提。代价是占住物理内存，所以别把所有东西都 pinned。

---

## 第 4 页｜Why Pass N?（为什么要传 N）

🎤 一个初学者必问的问题：数组多长你自己不知道吗，为什么非要把 N 传进 kernel？蓝框是书里的回答，意译：**CUDA kernel 被设计为在单个线程内部工作，与成千上万个其他线程并肩，各自处理输入数据的一个分片——N 定义了这个分片的边界**。
展开成三点：① CPU 函数可以自己查容器的长度（vector 有 size()）；kernel **做不到**——它拿到的只是一个**裸指针**，指针不携带长度信息，必须有人告诉它"世界的边界在哪"；② N 与 `blockIdx/blockDim/threadIdx` 配合，让一百万个元素被**不重、不漏、干净地**并行处理，横跨多个 SM；③ 这就是从 CPU 到 GPU 编程**最核心的思维转变**：你写的不是"处理这个数组"（一个循环），而是"处理**属于我的**那个元素——如果它存在的话"（一个判断）。

📖 页面对照：
- 蓝框（书中原意）："A CUDA kernel works inside of a single thread, on a partition of the input data. N defines the size of the partition." → kernel 在单个线程内部、在数据的一个分片上工作；N 定义分片边界。
- "you don't write 'process the array,' you write 'process my element --- if it exists.'" → 不写"处理数组"，写"处理我的元素——如果它存在"。

大白话：**一百万份复印的岗位说明书内容全一样；N 是上面写"活儿到哪里为止"的那一行。**

---

## 第 5 页｜The Bounds Check — and Lazily Surfacing Errors（边界检查与懒惰报错）⭐

🎤 上一页说"如果它存在的话"，这页把这件事彻底讲清，分四步。
**第一步：多出来的线程从哪来。** 蓝框的启动配置：`myKernel<<<1, 64>>>(input, 63)`——数组 63 个元素（下标 **0–62**），但 GPU 启动了 64 个线程（threadIdx.x = 0–63），组成 **2 个 warp**（注意 CUDA 从 **Warp 0** 开始编号）：**Warp 0** 管线程 0–31，全部有效；**Warp 1** 管线程 32–63——其中 32–62 有效，**线程 63 就是超出有效范围的那一个**。对它来说 `idx = 63`、`N = 63`，`63 < 63` 为 false——它不碰数组。
**第二步：这个 if 会不会让整个 warp 都不执行？不会。** Warp 1 里 lane 0–30（idx = 32–62）照常做乘法，只有 **lane 31 被 active mask 屏蔽**。一个 warp 按同一条指令执行，GPU 用活跃掩码控制哪些通道真正写数据——这里确实有一点分支分化，但只发生在最后一个 warp，通常可忽略（呼应 Part I 的分歧页）。
**第三步：不写检查会怎样？是未定义行为，不是"必然报错"。** 线程 63 会执行 `input[63] *= 2.0f`，而最后一个合法元素是 `input[62]`。后果三选一：**可能**报 `cudaErrorIllegalAddress`；**可能暂时什么错都不报**；**甚至可能悄悄破坏别的数据**——因为 `input[63]` 虽然超出数组的逻辑范围，却可能仍落在 GPU **已映射的内存页**里，硬件只判断"这个地址可不可访问"，它不知道"这个数组只到 62"。这就是越界 bug 难排查的原因：有时能跑，换个数据大小就炸。
**第四步：为什么 CPU 不会立刻知道。** kernel launch 默认**异步**：CPU 提交启动命令后不等 GPU 干完就继续往下跑；真正的非法访问发生在 GPU 执行期间，而 CPU 要到**下一次同步或某个 CUDA API 调用**时才看到错误——这就是标题里的 **Lazily Surfacing（延迟暴露）**。看右边的时间线代码：错误真正发生在 launch 那行，却在 `cudaDeviceSynchronize()` 那行才被报告——**最容易冤枉人的地方**：看起来是 sync 出错了，其实它只是替前面的异步 kernel"收到了错误通知"。
实践对策：调试期在每次 launch 后紧跟 `cudaGetLastError()` + `cudaDeviceSynchronize()`，把错误按在案发现场。

📖 页面对照：
- "Warp 1 (threads 32–63): thread 63 is the one extra beyond 0–62." → Warp 1 含线程 32–63；线程 63 是超出有效范围 0–62 的那一个。
- "lanes 0–30 of warp 1 execute, lane 31 is masked (active mask)" → warp 1 照常运行，仅 lane 31 被活跃掩码屏蔽。
- "input[63] is undefined behavior — may raise cudaErrorIllegalAddress, may silently pass or corrupt data (the address may still be mapped)" → 越界是未定义行为：可能报错、可能无声通过、可能破坏数据（地址可能仍在已映射页内）。
- "The sync didn't fail — it delivered the news: errors surface lazily." → sync 没有出错，它只是送信人；错误延迟浮现。

大白话：**GPU 出事不会给你打电话——它留张字条，你下次开信箱才看到。**

❓ 可能被问："生产代码也要每个 launch 都加 sync 吗？"→ 不要——`cudaDeviceSynchronize` 会杀性能。调试时加；生产上靠 `CUDA_LAUNCH_BLOCKING=1` 环境变量临时复现，或用 compute-sanitizer（Part V 讲）。
❓ "既然硬件不查数组边界，谁能查？"→ compute-sanitizer 的 memcheck 工具——它以调试模式跑 kernel，逐访问核对合法性，越界必抓（代价是慢，只在排查时用）。

---

## 第 6 页｜Configuring Launch Parameters: The Recipe in Code（launch 参数：代码配方）

🎤 "为什么 256"在 Part I 的实例页已经讲透，这页只兑现代码，**30 秒**：`threadsPerBlock = 256`；`blocksPerGrid = (N + 255) / 256`——整数除法配上"+255"就是**向上取整**，保证最后不满的一段也有 block 覆盖。技巧可推广：任意 block 大小 B，写 **`(N + B - 1) / B`**。最后一个 block 可能**不满员**——这正是第 5 页边界检查存在的原因，前后呼应一下。书里对 Blackwell 的提示：**256–512** 都可考虑；把 256 当**起点**而不是答案，正式跑之前用 profiling 对比 128/512。

📖 页面对照：
- "integer division rounds up, so every element is covered" → 向上取整，每个元素都被覆盖。
- "Treat 256 as the starting point, not the answer: profile 128 / 512 variants before committing." → 256 是起点不是答案，用 profiling 验证再定。

大白话：**256 是 block 尺寸里的"中杯咖啡"——第一次点单几乎不会错，profiling 说要改再改。**

---

## 第 7 页｜2D and 3D Kernel Inputs（二维与三维输入）

🎤 前面全是一维数组，处理图像、矩阵怎么办？代码只多三处：① 索引算**两次**——`x` 用 `.x` 系变量、`y` 用 `.y` 系变量，各自"blockIdx × blockDim + threadIdx"；② 边界检查变**二维**——`if (x < width && y < height)`；③ 访问前**拍平**——`idx = y * width + x`，二维坐标折回一维内存（显存里本来就是一维的）。
主机侧用 **`dim3`** 类型：`dim3 threads(16,16)` ——16×16 = 还是 256 个线程，只是排成方块；`dim3 blocks((W+15)/16, (H+15)/16)`——两个方向各自向上取整。
右侧要点：① 天然适合**图像**：比如 1,024×1,024 的矩阵用 16×16 的 block 铺满；② 同一套模式用 `dim3(x, y, z)` 推广到 **3D**——体数据（视频、医学影像）直接映射到线程层级；③ 书中注：本书大多数场景用 **1D 或 2D（分块）** 启动；纯 1D 时**普通 int 比 dim3 顺手**，不必为了形式统一硬上 dim3。

📖 页面对照：
- "Natural fit for images: a 1,024 × 1,024 matrix processed by 16 × 16 blocks (still 256 threads)." → 图像的天然搭配；16×16 依然是 256 线程。
- "volumetric data maps directly onto the thread hierarchy" → 体数据直接映射到线程层级。
- book note: "in 1D, plain ints beat dim3" → 一维场景普通 int 更好用。

大白话：**同一份菜谱，两个坐标轴：每个工人领到的是（行，列）工牌，而不是单个行号。**

🗣 **主动讲一下 4 维及更高维（batch, channel, height, width）怎么办**：CUDA 的 grid/block 最多只有 3 维，所以 ≥4 维张量横竖都要线性化——**逐元素操作的标准做法就是全部拍平当 1D**：`total = N×C×H×W`，一个线程管一个元素，线程根本不需要关心自己是哪个 batch（真要坐标就用除法/取模从 idx 反解）。那为什么还留着 2D/3D？两个理由：① 对图像/体数据**写法更自然**；② 更重要的是 **2D block 买到了空间局部性**——卷积、stencil、转置这类要访问**邻居**的操作，16×16 的 block 对应图像上一个 16×16 邻域，可以整块搬进共享内存复用（tiling）；拍平之后"邻居"就散了。准则一句话：**逐元素 → 拍平；要访问邻域 → 2D/3D**。
再补一个实战细节：结构化操作（batched matmul、attention）里，工程上经常拿 **grid 的 y/z 维来编码 batch 和 head**——比如 FlashAttention 的 grid 就是（M 方向块数, head 数, batch 数）。这时 3D grid 的作用是"给每个 (batch, head) 发一个独立班组"的便捷索引，不是空间含义。

❓ 可能被问："为什么是 16×16 不是 256×1？"→ 对二维数据，方形 block 让相邻线程访问相邻像素，访存局部性更好（第 7 章 coalescing 细讲）；对一维数据 256×1 就是标准答案。

---

## 第 8 页｜Allocate Asynchronously: Streams and Memory Pools（异步分配：流与内存池）⭐

🎤 到目前为止我们用的 `cudaMalloc`/`cudaFree` 有个隐藏的大坑，右栏第一条：它们是**同步且昂贵的**——每次调用要做**全设备同步**（等 GPU 上所有活干完！），还要走操作系统调用（`mmap`/`ioctl`）、陷入内核态再切回来。训练循环里每个迭代分配释放几次，这个开销就把你拖死了。
解法是左边六行代码：① `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)` 建一条**非阻塞流**——"流"就是一条 GPU 命令队列，同一条流内按序执行，流与流之间可以并行；**非阻塞**是为了避开老式**默认流的隐式全局栅栏**（书中提示；多流重叠第 11 章展开）；② **`cudaMallocAsync(&d_buf, size, s)`**——从**每设备的内存池**里取一块，不走操作系统；③ kernel 照常发进流 `s`；④ **`cudaFreeAsync(d_buf, s)`**——把"释放"作为一个命令排进流里，**只等流 s 自己干完**，不做全局 `cudaDeviceSynchronize`、不跨流同步。
内存池的三个好处（右栏第二条）：缓冲区**循环利用**、**没有操作系统往返**、长训练循环下**碎片更少**。

📖 页面对照：
- "cudaMalloc/cudaFree are synchronous and expensive: full-device sync + OS calls (mmap/ioctl), kernel-space context switches" → 同步且昂贵：全设备同步 + 系统调用 + 内核态切换。
- "cudaMallocAsync/cudaFreeAsync draw from a per-device memory pool: recycled buffers, no OS round trips, less fragmentation over long training loops" → 从每设备内存池取用：复用、免系统调用、少碎片。
- "Free waits only for stream s --- no global cudaDeviceSynchronize, no cross-stream sync." → 释放只等自己这条流。

大白话：**在停车场养一支车队、用时拿钥匙——别每次送货都买车再卖车。**

❓ 可能被问："流（stream）到底是什么？"→ 一条 GPU 命令队列：队内按序、队间可并行。你之前没写过流也一直在用——不指定就进"默认流"。

---

## 第 9 页｜Tuning the Pool; PyTorch's Caching Allocator（池调优与 PyTorch 分配器）——60 秒预告页

🎤 先定位：**这页是预告，不是深潜**——内存池的完整展开在**第 11 章**（stream-ordered allocation 有一整章），而且**第 12 章（我下次讲）**还会在 CUDA Graphs 里再遇到它。今天只要一个比喻加一个钩子。
**比喻（把上一页的池讲透）**：内存池就是**食堂的餐盘架**。没有池：每顿饭买个新盘子、吃完扔掉——每次分配释放都惊动操作系统。有池：盘子洗洗放回架子，下一个人直接拿——快，也不会越用越碎。左框两个旋钮就是食堂经理的两个决定：**架子上囤多少盘子才开始处理掉一些**（`cudaMemPoolAttrReleaseThreshold`：囤得多取用快、但占地方）；**现在立刻清一批**（`cudaMemPoolTrimTo`）。权衡就一句：**总占用 vs 碎片**。
**钩子（右框，大家最有感的部分）**：**PyTorch 就是一个自带餐盘架的食堂**——你每次建 tensor，它从自己的缓存分配器拿，根本不调 `cudaMalloc`。三个日常现象一次解释清：`nvidia-smi` 显存为什么比模型实际用量大（架子上囤着空盘子）；OOM 报错里 reserved 为什么远大于 allocated（同理）；`PYTORCH_ALLOC_CONF` 是干嘛的（给食堂经理下指令）。
**底部结论**：一次性大缓冲（如模型权重）用阻塞的 `cudaMalloc` 没问题；高频周转的循环（训练！）用异步 + 池，更稳更快。
收尾转场：分配的事到此为止，细节留给第 11 章。下一个问题——分配出来的内存**住在哪、访问代价是什么**？这就是 Part III。

📖 页面对照：
- 顶部灰字 "Preview only --- Chapter 11 is the deep dive; pools return in Chapter 12 (CUDA Graphs)." → 本页仅预告；第 11 章深潜，第 12 章在 CUDA Graphs 中重逢。
- "how much reserved memory the pool keeps before releasing to the OS" → 池保留多少内存才还给操作系统。
- "Trade-off: total footprint vs. fragmentation." → 权衡：总占用 vs 碎片。
- "PyTorch's caching allocator is the same idea: reuse GPU memory, avoid a synchronous cudaMalloc per new tensor." → PyTorch 缓存分配器同理：复用显存，避免每个新张量都同步分配。

大白话：**内存池 = 食堂餐盘架；PyTorch = 自带餐盘架的食堂——`nvidia-smi` 里"虚高"的显存就是架子上囤着的空盘子。**

❓ 可能被问："碎片化怎么治？"→ PyTorch 侧可试 `PYTORCH_ALLOC_CONF=expandable_segments:True`（可扩展段，显著缓解长训练的碎片 OOM）；CUDA 侧靠池本身的复用 + TrimTo。
