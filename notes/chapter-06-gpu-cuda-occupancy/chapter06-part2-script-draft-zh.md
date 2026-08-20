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

🎤 上一页说"如果它存在的话"，这一页讲不检查存在性的下场。左边蓝框是算例：**N = 63**。调度器按 warp 分配线程，63 个线程要 **2 个 warp = 64 个线程**。warp 1 处理元素 0–31，没问题；warp 2 该处理 32–62，但它有 32 个线程，**多出 1 个**——没有 `if (idx < N)` 的话，第 64 个线程会去读数组外面 → **`cudaErrorIllegalAddress`**。书里的规矩：边界检查在 CUDA 代码里无处不在；**"如果你在某段代码里没看到它，先搞清楚它为什么不在。"**
右边讲报错机制，这是 CUDA 新手最大的坑之一：① kernel **异步执行，没有线程级异常**——GPU 上没有 try/catch，一次非法访问只是给整个 launch 设置一个**全局故障标志**；② 主机要等到**下一次同步或下一次 CUDA API 调用**才能看到这个错——错误是**延迟浮现（lazily surfacing）**的，报错位置往往不是案发现场；③ 实践对策：调试时在每次 launch 后面紧跟 **`cudaGetLastError()` + `cudaDeviceSynchronize()`**，把错误按在案发现场抓住。

📖 页面对照：
- worked example N=63: "Warp 2 expects 32--63 plus one extra thread; without if (idx < N) it reads past the array" → warp 2 多出一个线程，不检查就越界。
- "kernels run asynchronously, with no per-thread exceptions: an illegal access sets a global fault flag for the whole launch" → 异步执行、无线程级异常；非法访问设全局故障标志。
- "errors surface lazily" → 错误延迟浮现。

大白话：**GPU 出事不会给你打电话——它留张字条，你下次开信箱才看到。**

❓ 可能被问："生产代码也要每个 launch 都加 sync 吗？"→ 不要——`cudaDeviceSynchronize` 会杀性能。调试时加；生产上靠 `CUDA_LAUNCH_BLOCKING=1` 环境变量临时复现，或用 compute-sanitizer（Part V 讲）。

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

## 第 9 页｜Tuning the Pool; PyTorch's Caching Allocator（池调优与 PyTorch 分配器）

🎤 内存池不是免费午餐，有两个旋钮（左框）：① **`cudaMemPoolAttrReleaseThreshold`**（通过 `cudaMemPoolSetAttribute` 设置）——池子攒到多少保留内存才开始还给操作系统：调高，分配更快但显存占用虚高；调低，省显存但又要频繁走系统调用；② **`cudaMemPoolTrimTo`**——手动"修剪"，主动归还内存。核心权衡就一句：**总占用量 vs 碎片化**。
右框是和大家日常最相关的连接点：**PyTorch 的缓存分配器就是同一个思想**——你每次 `torch.empty(...)` 并没有真的走 `cudaMalloc`，PyTorch 自己维护着一个缓存池，环境变量 **`PYTORCH_ALLOC_CONF`**（旧名 `PYTORCH_CUDA_ALLOC_CONF`）就是它的旋钮。所以：为什么 `nvidia-smi` 显示的显存占用比模型实际需要的大？为什么有时报 OOM 但"reserved"远大于"allocated"？——都是这个池子在起作用。
底部是书里的结论：**一次性的大缓冲区，用阻塞的 `cudaMalloc` 完全没问题；分配密集的循环（训练！），异步 + 内存池的性能更稳、吞吐更高。**

📖 页面对照：
- "how much reserved memory the pool keeps before releasing to the OS" → 池子保留多少内存才还给操作系统。
- "Trade-off: total footprint vs. fragmentation." → 权衡：总占用 vs 碎片。
- "PyTorch's caching allocator is the same idea: reuse GPU memory, avoid a synchronous cudaMalloc per new tensor per iteration." → PyTorch 缓存分配器同理：复用显存，避免每个迭代每个新张量都同步分配。
- verdict: "one-off buffers --- blocking cudaMalloc is fine; allocation-heavy loops --- async + pools." → 一次性缓冲用同步分配即可；分配密集循环用异步 + 池。

大白话：**你在 PyTorch 里其实天天在用这一页——`nvidia-smi` 里"虚高"的显存就是那支停在场里的车队。**

❓ 可能被问："碎片化怎么治？"→ PyTorch 侧可试 `PYTORCH_ALLOC_CONF=expandable_segments:True`（可扩展段，显著缓解长训练的碎片 OOM）；CUDA 侧靠池本身的复用 + TrimTo。
