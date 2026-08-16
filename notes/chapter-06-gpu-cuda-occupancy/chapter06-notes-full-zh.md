# 第六章精读笔记（完整版）：GPU Architecture, CUDA Programming, and Maximizing Occupancy

> **对应原文**：书印刷页码 191–238（共 48 页；即切出的 `chapter06.pdf` 第 1–48 页，换算：PDF 页 = 印刷页 − 190）。
> **本笔记的写法**：严格按原书小节顺序逐节精读。每节包含：内容详解（中文）、代码逐行解释、书中表格完整复刻、插图说明、边栏提示（书里灰色的 Note/Tip 框，很多实战干货藏在这里）、关键英文原句对照。
> **快速版笔记**见 `chapter06-notes-zh.md`；**逐页讲稿**见 `chapter06-lecture-script-zh.md`。

---

## 目录（对照原书结构）

| 原书小节 | 印刷页 | 本笔记 |
|---|---|---|
| 章首引言 | 191 | §0 |
| Understanding GPU Architecture | 191–202 | §1 |
| ├ Threads, Warps, Blocks, and Grids | 195–200 | §1.3 |
| ├ Choosing Threads-per-Block and Blocks-per-Grid Sizes | 200–202 | §1.4 |
| └ CUDA GPU Backward and Forward Compatibility Model | 203 | §1.5 |
| CUDA Programming Refresher | 203–230 | §2 |
| ├ 第一个完整 kernel | 203–206 | §2.1 |
| ├ Configuring Launch Parameters | 207–209 | §2.2 |
| ├ 2D and 3D Kernel Inputs | 209–210 | §2.3 |
| ├ Asynchronous Memory Allocation and Memory Pools | 210–212 | §2.4 |
| ├ Understanding GPU Memory Hierarchy | 212–217 | §2.5 |
| ├ Unified Memory | 218–221 | §2.6 |
| ├ Maintaining High Occupancy and GPU Utilization | 221–229 | §2.7 |
| └ Tuning Occupancy with Launch Bounds | 229–231 | §2.8 |
| Debugging Functional Correctness with NVIDIA Compute Sanitizer | 231–232 | §3 |
| Roofline Model | 233–236 | §4 |
| Key Takeaways | 237 | §5 |
| Conclusion | 238 | §6 |

---

# §0 章首引言（p.191）

本章路线图（原文第一段的承诺）：
1. 复习 **SIMT 执行模型**——warp、thread block、grid 如何把你的算法映射到 SM 上；
2. **CUDA 编程模式**；
3. **片上内存层级**——寄存器文件、共享内存/L1、L2、HBM3e；
4. GPU 的**异步数据传输**能力——包括 **TMA**（Tensor Memory Accelerator，张量内存加速器）和 **TMEM**（Tensor Memory，作为 Tensor Core 运算的**累加器**的专用内存）；
5. **Roofline 分析**——区分 compute-bound（算力受限）和 memory-bound（访存受限）的 kernel。

> 原句：**"This will provide the fundamentals to push modern GPU systems toward their theoretical peak throughput ceilings."**
> 这些是把现代 GPU 系统推向理论峰值吞吐天花板所需的基本功。

---

# §1 Understanding GPU Architecture（理解 GPU 架构，p.191–202）

## §1.1 GPU 的设计哲学与基本协作流程（p.191–192）

**核心对比**：CPU 为"低延迟的单线程性能"优化；GPU 是"吞吐量优化"的处理器，为并行运行**几千个线程**而造。

**Figure 6-1（简单 CUDA 编程流程）** 描述了最基本的 CPU-GPU 协作四步：
1. host（主机，CPU 侧）把数据装载进 CPU 内存；
2. 把数据从 CPU 内存拷到 GPU 显存；
3. 用显存里的数据调用 GPU kernel；
4. CPU 把结果从 GPU 显存拷回 CPU 内存，继续后处理。

GPU 靠**海量并行来隐藏数据传输延迟**（包括上述 CPU-GPU 拷贝）。每块 GPU 由许多 **SM（Streaming Multiprocessor，流式多处理器）** 组成——粗略类比 CPU 的核，但为并行做了精简（"streamlined for parallelism"）。

**Blackwell 每 SM 的关键数字**（p.192，务必记住）：
- 每 SM 最多跟踪 **64 个 warp**（64 × 32 = **2048 个线程**）并发；
- **64K 个 32 位寄存器**（共 256 KB）；
- **256 KB 统一 L1 缓存/共享内存**，其中最多 **228 KB** 可配置为用户管理的共享内存（**实际可用 227 KB**——CUDA 从 228 KB 里保留 1 KB）；单个 thread block 最多申请 227 KB 动态共享内存。

## §1.2 SM 内部：四个 warp 调度器与双发射（p.192–194）

**Figure 6-2**：Blackwell 的 SM 内含**四个独立 warp 调度器**（warp scheduler），每个每拍（时钟周期）可发射一个 warp 的指令。

> 原句：**"You can think of the SM as four 'mini-SMs' sharing on-chip resources."**
> 可以把 SM 想成四个共享片上资源的"迷你 SM"。

两层并行机制：
1. **跨 warp**：4 个调度器 → 每拍最多 **4 个 warp** 同时推进（前提：有足够的独立工作和可配对指令）。
2. **warp 内双发射（dual-issue）**：每个调度器每拍可以从**同一个 warp** 发出两条指令——一条算术（INT32 / FP32 / Tensor Core）+ 一条访存（load/store）。
   > 原句：**"Note that the dual-issue must come from the same warp—and not across warps."**
   > 双发射必须来自同一个 warp，不能跨 warp。

**Table 6-1：SM 调度器与指令发射上限（每时钟周期）**

| 指标 | 值 |
|---|---|
| 调度器数量 | 4 |
| 最多发射 warp 数 | 4（每调度器 1 个）|
| 最多算术操作 | 4（每调度器 1 条算术发射）|
| 最多访存操作 | 4（每调度器 1 条 load/store 发射）|

📦 **边栏 Note（p.193，全书通用免责声明）**：
> "The numeric values in all metrics tables are illustrative to explain the concepts. For actual benchmark results on different GPU architectures, see the GitHub repository."
> **所有指标表中的数值都是为讲解概念的示意值**。不同 GPU 架构的真实 benchmark 见配套 GitHub 仓库。
> （被问到"这些数字准吗"时用这条回答。）

所以最好情况是：**每拍跨 4 个 warp 双发射 4 条数学 + 4 条访存指令**——同时拉满计算和访存吞吐。

**SFU（Special Function Unit，特殊功能单元）**（p.194）：
- 与 INT32、FP32、Tensor Core 管线并列，处理**超越函数**（sine、cosine、倒数 reciprocal、平方根 square root）；
- **不占双发射的"数学+访存"名额**——它有自己独立的 SFU 管线；
- 意义：慢速复杂运算不会堵住核心的计算和访存管线，**进一步提高指令级并行（ILP）**，对混合运算的 kernel 尤其有利。

**LD/ST（load/store，访存）管线**（p.194）：访存操作流经 SM 的 **16 条 LD/ST 管线**（每调度器 4 条），读写 L1/共享内存、L2 或全局显存。

📦 **边栏 Warning（p.194）**：
> "Exact LD/ST pipeline counts and pairings are not guaranteed. Rely on profiling counters... consult the NVIDIA documentation... The Blackwell tuning guide is a good place to start."
> **确切的 LD/ST 管线数量和配对规则不受保证**。要靠 profiling 计数器判断 kernel 是受限于访存发射还是计算发射；具体架构细节查 NVIDIA 文档，Blackwell tuning guide 是好起点。

本节小结：GPU 擅长**数据并行**工作负载——大矩阵乘、卷积等"同一条指令作用于海量元素"的运算。开发者可以直接写 CUDA C++，也可以间接经由 PyTorch 或 Python 系的 GPU 语言（如 OpenAI Triton）。

## §1.3 Threads, Warps, Blocks, and Grids（线程、warp、块、网格，p.195–200）

### 三层层级（Figure 6-3 / 6-4）

CUDA 把并行工作组织成三层，**平衡"可编程性"与"海量吞吐"**：
- **Thread（线程）**：最底层，每个线程执行你的 kernel 代码；
- **Thread block（线程块，又名 CTA，cooperative thread array，协作线程阵列）**：最多 **1024 线程**一组；
- **Grid（网格）**：kernel 启动时全部 block 的集合。

> 原句：**"By sizing your grid appropriately, you can scale to millions of threads without changing your kernel logic."**
> 把 grid 尺寸设对，你可以扩展到几百万线程而不改一行 kernel 逻辑。CUDA 运行时（以及 PyTorch 这类框架）负责调度、把 block 分发到所有 SM。

Figure 6-4 补充了 host 视角：CPU 上的 host 代码发起 kernel 调用，kernel 运行在 GPU device 上。

### Thread Block Cluster 与 DSMEM（p.196–197，Figure 6-5）

传统规则：**不同 block 的线程不能直接协作**。现代架构（Hopper/Blackwell）+ 新版 CUDA 打破了这一点：
- **Thread block cluster（线程块簇）**：一组可以**跨 SM 相互通信**的 thread block；
- 簇内不同 block 的线程可以**访问彼此的共享内存**，并使用**硬件支持的簇级 barrier**；
- 底层硬件是 **DSMEM（distributed shared memory，分布式共享内存）**：把参与簇的各 SM 的共享内存 bank 通过**片上高速互连**连成一个统一地址空间——多 SM 合用一个"分布式共享内存池"。

> 原句：**"This unification allows threads in different blocks to read, write, and atomically update one another's shared buffers at on-chip speeds—and without using global memory bandwidth."**
> 这种统一让不同 block 的线程能以**片上速度**读、写、原子更新彼此的共享缓冲区——**不占用全局显存带宽**。

📦 **边栏 Note（p.197）**：cluster 和 DSMEM 在**第 10 章**详讲；它们是现代 GPU 极重要的新增能力（大矩阵乘、LLM 负载的关键）。本章聚焦 **block 内**的共享内存优化。

### Block 内同步（p.197，Figure 6-6）

- Block 内线程通过低延迟片上**共享内存**交换数据，用 **`__syncthreads()`** 同步（栅栏：所有线程到齐才继续）。
- **每个 barrier 都有开销 → 尽量减少同步点。**
- 同时，GPU 硬件会通过**快速切换 warp** 来隐藏长延迟事件（全局显存加载、缓存填充、管线停顿）。

### Warp 与 SIMT（p.198，Figure 6-7）

- Thread block 再被细分为 **32 线程的 warp**，在 **SIMT**（single instruction, multiple threads）模型下由 warp 调度器管理、**锁步（lockstep）执行**。
- **Occupancy（占用率）** 首次正式定义：让更多 warp "在飞"（in flight）就是 SM 上的高占用率。代码允许高占用率时，一个 warp 停了，另一个随时能跑——保持计算单元忙碌。
- **制衡**：高 occupancy 必须与**每线程资源上限**（寄存器、共享内存）平衡。**寄存器溢出（spilling）到慢速内存会制造新的停顿。** 应对：把 occupancy 和寄存器/共享内存用量放在一起 profile，选一个不引发资源争抢的 block size。

📦 **边栏 Note（p.198）**：occupancy 调优在**第 8 章**展开，但它是理解 SM/warp/线程的核心概念。

- **Block 独立性**：thread block 之间独立执行、顺序无保证 → GPU 调度器可以把它们撒到所有 SM 上充分利用硬件 → **你的 kernel 在未来更多 SM 的架构上不改就能跑**（grid–block–warp 层级的可移植性保证）。

### Warp Divergence（warp 分叉，p.199–200，Figure 6-8）

吞吐还取决于 **warp 执行效率**：同一 warp 内的线程要走**同一条控制流路径**、做**合并的（coalesced）访存**。

分叉机制：warp 内部分线程走 `if`、部分走 `else` 时，warp **串行化**执行——逐条路径处理：
> 原句：**"By masking inactive lanes and running extra passes to cover each branch, warp divergence multiplies the overall execution time by the number of branches."**
> 通过屏蔽（mask）不活跃车道（lane）、跑额外遍数覆盖每个分支，warp 分叉把总执行时间**乘以分支数**。

📦 **边栏 Note（p.200）**：
> "Divergence is an issue only for threads within a single warp. Different warps can follow different branches with no performance penalty."
> 分叉**只**是单个 warp 内的问题。**不同 warp 走不同分支没有性能惩罚。**

检测、profile、缓解手段 → 第 8 章。

## §1.4 选择 Threads-per-Block 与 Blocks-per-Grid（p.200–202）

**为什么必须 32 的倍数**：block size 要对齐硬件的 32 线程 warp 尺寸。例：256 线程 block = 8 个整 warp；而 **33 线程的 block 要占两个 warp 槽位、第二个 warp 只用了 1/32 的车道**——每个 warp 不管跑 32 个还是 1 个线程都占一个调度器槽位，纯浪费并行机会。

**代际差异限制 block 大小**：不同代 GPU 的每 SM 最大线程数、寄存器数不同。block 太大 → 寄存器不够 → **register spilling** → 性能下降；block 太大也可能要太多共享内存（Blackwell 每 SM 只有 228 KB/227 可用，由 SM 上全部常驻 block 共享）。

这些硬件上限决定了一个 SM 上能同时活跃多少 block/warp——即 occupancy。**小一点的 block 可能带来更高的 occupancy**（能塞进更多并发 warp）。

**Figure 6-9**：线程资源的相对规模（thread → warp → block → SM → GPU 的数量级阶梯）。

**Table 6-2：线程级与块级上限（Blackwell B200）**

| 资源 | 硬件上限 | 备注 |
|---|---|---|
| Warp size | 32 线程 | SIMT 基本执行单位；**永远用 32 的倍数**避免浪费 |
| 每 block 最大线程数 | 1,024 | `blockDim.x × blockDim.y × blockDim.z ≤ 1024` |
| 每 block 最大 warp 数 | 32 | 1024 ÷ 32 = 32 |

**Table 6-3：SM 常驻（SM-resident）资源上限（Blackwell B200）**

| 资源（每 SM） | 上限 | 备注 |
|---|---|---|
| 最大常驻 warp | **64** | 64 × 32 = 2048 线程。**该上限已保持多代不变**，occupancy 经验可以延续 |
| 最大常驻线程 | **2,048** | 若每 block 1024 线程 → 每 SM 最多 2 个 block；**若 256 线程/block → 可驻 8 个 block**（8×256=2048），occupancy 更高、更好藏延迟——但过多迷你 block 会增加调度开销 |
| 最大活跃 block | **32** | block 更小则可以塞到这个上限 |

**Table 6-4：CUDA grid 上限**

| 维度 | 上限 | 备注 |
|---|---|---|
| X | 2,147,483,647 个 block | 3D grid 可达 2,147,483,647 × 65,535 × 65,535 |
| Y / Z | 各 65,535 | |
| 并发 grid（kernel）数 | **128** | 一台设备最多 128 个 kernel 并发执行 |

实践提示：**你几乎总是先撞上 thread/block/SM 限制，而不是 grid 限制**。若某维需要超过 65,535 个 block，可用 2D/3D grid 或多次 kernel 启动（multilaunch）拆分。

## §1.5 CUDA 的前后向兼容模型（p.203）

CUDA 生态的核心优势之一：**今天编译的 kernel 一般能不改跑在未来的 GPU 上——前提是二进制里带 PTX**。

关键概念：
- **SASS**：特定架构的最终机器码（如 sm_90 = Hopper、sm_100 = Blackwell）。**只带单架构 SASS、不带 PTX 的二进制，不能在更新的架构上运行。**
- **PTX**：虚拟指令集（中间表示），驱动可以在加载时 JIT（即时编译）成新架构的 SASS → 前向兼容的保障。
- **家族目标**（如 `sm_100f` / `compute_100f`）：可移植性被限制在**同特性家族**的设备内。
- **最佳实践：发 fatbin（胖二进制）**——同时打包通用 cubin/PTX + 所需的家族专用 cubin（为了架构特定优化）。

**验证方法**：设 `CUDA_FORCE_PTX_JIT=1` 强制加载时走 PTX JIT 编译并缓存结果。**如果二进制缺 PTX，kernel 启动直接失败**——逼你重新带 PTX 构建。

📦 **边栏 Tip（p.203）**：要真正保持跨代兼容，用**通用目标**编译或显式包含 PTX；需要新硬件特性的专门优化时，用代际专用目标，但**务必为其他架构提供 fallback 路径**。

---

# §2 CUDA Programming Refresher（CUDA 编程速览，p.203–230）

## §2.1 第一个完整 kernel：逐行精读（p.203–206）

**语法要素**：kernel 是 `__global__` 标注的特殊函数，跑在 GPU device 上；从 CPU host 代码调用时用 **`<<< >>>`"三尖括号"（chevron）语法**指定两个配置参数：`blocksPerGrid`（块数）和 `threadsPerBlock`（每块线程数）。

书中第一个完整例子（把数组每个元素原地翻倍）：

```cuda
// ---------- device 侧 ----------
__global__ void myKernel(float* input, int N) {
    // 计算全局唯一线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 只处理有效元素（边界检查）
    if (idx < N) {
        input[idx] *= 2.0f;
    }
}

// ---------- host 侧 ----------
int main() {
    const int N = 1'000'000;              // 一百万个 float
    float *h_input = nullptr;             // h_ 前缀 = host 内存
    float *d_input = nullptr;             // d_ 前缀 = device 显存
    cudaMallocHost(&h_input, N * sizeof(float));   // ① 分配 pinned host 内存
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f; // ② 初始化
    cudaMalloc(&d_input, N * sizeof(float));       // ③ 分配显存
    cudaMemcpy(d_input, h_input, N * sizeof(float),
               cudaMemcpyHostToDevice);            // ④ H2D 拷贝
    const int threadsPerBlock = 256;               // ⑤ 32 的倍数
    const int blocksPerGrid =
        (N + threadsPerBlock - 1) / threadsPerBlock; // N=1M 时 = 3907
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N); // ⑥ 启动
    cudaDeviceSynchronize();                       // ⑦ 等 kernel 完成
    cudaMemcpy(h_input, d_input, N * sizeof(float),
               cudaMemcpyDeviceToHost);            // ⑧ D2H 拷回
    cudaFree(d_input); cudaFreeHost(h_input);      // 清理
    return 0;
}
```

**逐点讲解**：

1. **命名习惯**：`h_` = host 指针，`d_` = device 指针——全书通用，建议照抄。
2. **`cudaMallocHost`** 分配的是 **pinned（页锁定）内存**——不会被操作系统换页，是后面异步拷贝能真正重叠的前提（§4 会再次强调）。
3. **数据流六步**（书里明确列出）：分配 host 内存 → H2D 拷贝 → device 上跑 kernel → 同步确认完成 → D2H 拷回 → `cudaFree`/`cudaFreeHost` 清理。
4. `<<< >>>` 还能传更多高级参数（共享内存大小等），但 `blocksPerGrid` 和 `threadsPerBlock` 是任何 kernel 调用的根基。

📦 **边栏 Note（p.205）**：这段代码**没有完全优化**——全书会持续优化它。但它是你构建自己 kernel 的**简单、完整的模板**。

**为什么要传 N？**（p.205，很多初学者的疑问）
> 原句："a CUDA kernel function is designed to work inside of a single thread, alongside thousands of other threads, on a partition of the input data. As such, N defines the size of the partition that this particular kernel will process."
> CUDA kernel 函数的设计是**在单个线程内**工作、与几千个其他线程并肩、处理输入数据的**一个分区**。N 定义了这个 kernel 要处理的分区大小。

配合内置变量 `blockDim`（本例一维）、`blockIdx`、`threadIdx`，kernel 算出自己独有的 `idx`——让每个元素被**干净且唯一地**并行处理，跨越许多同时运行的 SM。

**边界检查 `if (idx < N)` 的完整推理**（p.206，N=63 的例子）：
- 输入数组 63 个元素 → 调度器大概率派 **2 个 warp**（各 32 线程）；
- 第一个 warp 处理元素 0–31，没问题；
- 第二个 warp "以为"要处理 32–64——**若没有边界检查，线程 idx=63 之后的（准确说 idx≥63 的那部分，例中 idx=64 及以上不存在，真正越界的是 idx=63 以外的 63、64 号线程位置）会试图访问越界地址**，抛出**非法内存访问**错误（如 `cudaErrorIllegalAddress`）；
- 有了检查，每个线程"要么处理有效元素、要么立即退出"。

**CUDA 错误的"懒惰上报"机制**（p.206，容易踩坑）：
> 原句："CUDA kernels execute asynchronously on the device without per-thread exceptions; instead, any illegal operation sets a global fault flag for the entire launch. The host driver only checks that flag when you next call a synchronization or another CUDA API function, so errors surface lazily."
> kernel 在 device 上**异步**执行，没有"每线程异常"；任何非法操作（越界、未对齐访问等）只是给整次 launch 设一个**全局故障标志**。host 驱动**只在你下次调用同步或别的 CUDA API 时**才检查这个标志——错误是**延迟浮现**的。

- 这个设计让 GPU 管线和互连保持满载，但要求你**在 host 侧显式同步并轮询错误**——通常在 kernel 启动后立刻 `cudaGetLastError()` + `cudaDeviceSynchronize()`，尽早抓住故障。
- 书中金句：**"You will see a bounds check in a lot of CUDA kernels. If you don't see it, you should understand why it's not there."**——大量 kernel 都有边界检查；如果没看到，你应该弄明白它为什么不在（要么以别的形式存在，要么作者能证明永远不会越界）。

## §2.2 Configuring Launch Parameters（配置启动参数，p.207–209）

`threadsPerBlock = 256`（8 个 warp）是**平衡 occupancy 与资源占用的常用起点**，四个理由（书里逐条给出）：

1. **Multiple of 32（32 的倍数）**：避免空 warp 槽位——欠填充的 warp 占着稀缺的调度器资源却不产出。
2. **Latency hiding（延迟隐藏）**：每 SM 需要几百个线程才能藏住 DRAM 和指令延迟。例如在容量 2048 线程的 SM 上放 8 个 256 线程的 block，正好喂满管线又不过度订阅。
3. **Occupancy**：256 线程 = 每 block 只要 8 个 warp，通常能有好的 occupancy，又不至于耗尽每 block 的寄存器或共享内存。
   📦 **边栏 Tip**：对 Blackwell 这类现代 GPU，考虑 **256–512 线程/block**，在尊重寄存器和共享内存上限的同时最大化 occupancy。
4. **Resource-balanced（资源均衡）**：256 足够小（远离 1024 上限），又足够大（别的 warp 停顿时不会闲太多 warp）。

从 256 出发，再根据 kernel 的寄存器/共享内存需求和 occupancy 特性上下调（128、512 等）。

`blocksPerGrid` 常用 **`(N + threadsPerBlock - 1) / threadsPerBlock`**——向上取整，保证 N 不是 block size 整数倍时**每个输入元素都有线程覆盖**。（书里附了第二个完整代码清单，同一个 kernel、动态计算两个参数，配合熟悉的边界检查——"多出来的"线程什么也不做、不会引发非法访问。）

## §2.3 2D 和 3D Kernel 输入（p.209–210）

数据天然是二维时（如图像），可以启动 **2D grid × 2D block**。书中例子：1024×1024 矩阵、16×16 的 block（正好 256 线程）：

```cuda
__global__ void my2DKernel(float* input, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // 列坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // 行坐标
    if (x < width && y < height) {                   // 2D 边界检查
        int idx = y * width + x;                     // 摊平成一维下标
        input[idx] *= 2.0f;
    }
}
```

- host 侧用 `dim3` 类型定义 `blocksPerGrid2D` / `threadsPerBlock2D`；同样的模式**用 `dim3(x, y, z)` 直接推广到 3D**，把体数据（volumetric data）直接映射到线程层级。
- 书中此例的 main 已经用上了下一节的进阶写法：非阻塞 stream + `cudaMallocAsync` + `cudaMemcpyAsync` + `cudaStreamSynchronize` + `cudaFreeAsync`——作者在悄悄示范"更好的默认习惯"。

📦 **边栏 Note（p.210）**：本书大部分场景用 1D 或 2D（tiled，分块）的启动参数；1D 时用普通常量即可，不必用 `dim3`。

## §2.4 异步内存分配与内存池（p.210–212）

**问题**：标准 `cudaMalloc`/`cudaFree` 是**同步且相对昂贵**的——
- 需要**全设备同步**（较慢）；
- 涉及 **OS 级调用**（`mmap`/`ioctl`）管理 GPU 内存 → 内核态上下文切换 + 驱动开销。

**方案**：用异步版本 **`cudaMallocAsync` / `cudaFreeAsync`**。
- CUDA 运行时默认维护一个**全局 GPU 内存池**；异步释放的内存回到池里等待后续分配复用；
- 内存池**回收已释放的缓冲区**、避免反复找 OS 要新内存 → 长训练循环里**减少碎片化**；
- **PyTorch 对应物**：自定义的 **caching allocator**（配置项 `PYTORCH_ALLOC_CONF`，旧名 `PYTORCH_CUDA_ALLOC_CONF`）——精神上与 CUDA 内存池一致：复用显存、避免每轮迭代为新 tensor 调用同步的 cudaMalloc。

**stream-ordered allocation（流序分配）三件套**：

```cuda
cudaStream_t stream1;
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);  // ① 非阻塞流

float* d_buf = nullptr;
cudaMallocAsync(&d_buf, N * sizeof(float), stream1);         // ② 流上分配
myKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_buf, N);
cudaFreeAsync(d_buf, stream1);   // ③ 释放被推迟到 stream1 的活干完
```

- 这些 API 从**每设备内存池**分配，但**尊重所传 stream 的顺序**——释放推迟到该 stream 工作完成；
- `cudaFreeAsync` 只等 stream1 → **没有昂贵的全局 `cudaDeviceSynchronize`、不与其他 stream 隐式同步**；
- 收益：当代码发出**成千上万乃至百万次**分配/释放循环时，分配开销大幅降低、碎片减少、延迟毛刺被抹平。

📦 **边栏 Tip（p.211）**：显式使用 CUDA stream 是重叠传输/kernel/内存操作的最佳实践。把每条 stream 看作**独立通道**，只对自己内部的操作强制排序。**推荐用 `cudaStreamNonBlocking` 建流，避开老式默认流（legacy default stream）的隐式全局屏障。** 多流重叠技巧详见第 11 章。

**内存池进阶调优**：
- `cudaMemPoolSetAttribute` + **`cudaMemPoolAttrReleaseThreshold`**：提示池子在尝试向系统归还内存前保留多少——权衡"总显存占用"与"碎片化"；
- **`cudaMemPoolTrimTo`**：主动归还内存。

**选型结论**：一次性的简单缓冲区，用阻塞的 cudaMalloc/cudaFree 也行；**长期运行、反复分配释放的复杂循环，切到专用 stream 上的 Async 版本 + 内存池**，性能更稳、吞吐更高。

## §2.5 GPU 内存层级（p.212–217）⭐全章最重要的一节

背景：前面讲的分配都来自 stream 的内存池（默认 stream 0 也有）、落在全局显存。实际上 GPU 提供**多级内存层级**平衡容量与速度：寄存器、共享内存、各级缓存、全局显存，以及 Blackwell 起新增的 **TMEM**。

**Table 6-5：Blackwell 内存层级与特性（完整版）**

| 层级 | 作用域 | 容量 | 延迟 | 带宽（约） |
|---|---|---|---|---|
| Registers（寄存器） | 每线程（在 SM 上） | 64K 个 32 位/SM（每线程最多 255 个） | 近似零附加延迟（读写单周期、基本免费） | 每 SM 几十 TB/s（寄存器堆端口） |
| Shared memory + L1 | 每 SM | 228 KB（可用 227）共享 + 其余作 L1/数据缓存 | ~20–30 周期 | 每 SM TB/s 级（无 bank 冲突时） |
| TMEM | 每 SM | 256 KB SRAM，专属 Tensor Core | ~10 周期（SM 上专用 SRAM） | 与 Tensor Core 之间 TB/s 级 |
| Constant memory cache（常量缓存） | 每 SM | ~8 KB 缓存，前置 64 KB `__constant__` 空间 | ~1 周期（warp 广播）；命中且全 warp 同地址时与寄存器一样快；分歧或未命中则串行化/延迟升高 | TB/s 级（广播吞吐） |
| L2 cache | 全 GPU（所有 SM） | **126 MB** | ~200 周期 | 聚合数十 TB/s |
| Local memory（局部内存） | 每线程（溢出到 DRAM） | 近乎无限（由全局显存背书） | 数百 → 1000 周期（DRAM 级） | ~8 TB/s（HBM3e） |
| Global memory（HBM/DRAM） | 全设备（片外 DRAM） | B200 最多 **180 GB**（B300 约 288 GB） | 数百 → 1000 周期 | **~8 TB/s** 总带宽 |

> 原句：**"maximizing data reuse in registers, shared memory, and L1/L2 cache—and minimizing reliance on global memory and local memory—is essential for high-throughput GPU kernels."**
> 把数据复用最大化地留在寄存器、共享内存、L1/L2，把对全局显存和局部内存的依赖降到最低，对高吞吐 kernel 是决定性的。

**逐层详解**（书中每层都有一段，含插图）：

**（1）寄存器（Figure 6-12 局部内存）**：每个线程的"起点"。单周期读写、几乎不与任何东西争抢，带宽可达每 SM 几十 TB/s。**但是**：kernel 需要的寄存器超标（线程局部变量太多或编译器临时量太多）时，**溢出（spill）进 local memory**——名字叫 local，物理上映射到**片外 DRAM**，代价是几百到一千多周期。这是"性能悬崖"。

**（2）共享内存与 L1 数据缓存（Figure 6-13）**：统一的 256 KB 每 SM 片上 SRAM，可在"用户管理的共享内存"（最多 228 KB/block 级 227 KB）和 L1 之间动态划分——用 `cudaFuncSetAttribute()` + `cudaFuncAttributePreferredSharedMemoryCarveout` 选择**carveout（划分比例）**。访问 20–30 周期；**把 thread block 设计成避开 bank conflict（存储体冲突），就能拿到 TB/s 级吞吐**。

**（3）TMEM（Figure 6-11）**：每 SM 256 KB 专用片上内存，供 Tensor Core 专属操作/指令（UMMA——unified matrix-multiply-accumulate、tcgen05，第 10 章详讲）。**不是 CUDA C++ 里普通的可寻址空间**——传输由 **TMA 用描述符（descriptor）编排**，开发者不用手工管理 Tensor Core 的数据流。以 C = A × B 为例：操作数 B 来自 SMEM，A 在 TMEM（也可在 SMEM），**累加器在 TMEM**；tile 从全局显存经 L2 由 TMA（如 `cuda::memcpy_async`）流入 SMEM；SMEM↔TMEM 之间由 Tensor Core 指令隐式搬运。TMEM 与 Tensor Core 之间以**每秒几十 TB** 级带宽透明通信，减少 Tensor Core 对全局显存的依赖。

**（4）常量缓存**：小的只读表的最佳去处。**全 warp 32 线程读同一地址 → 单周期广播**；分歧读会跨 lane 串行化。书中点名的适用场景（都是 LLM 高频小表）：**旋转位置编码（rotary positional encodings）查找表、ALiBi（Attention with Linear Biases）斜率、LayerNorm 的 γ/β 向量、embedding 量化 scale**——所有线程共享、零全局显存流量。

**（5）L2**：片上 SRAM 之外、连接所有 SM 与片外 HBM3e 的 **126 MB** 全 GPU 缓冲。延迟约 200 周期、聚合带宽几十 TB/s，吸收 L1 的溢出。**跨 block 复用**：一个 block 取来的数据，其他 block 不用重访 DRAM。
📦 **边栏 Tip（p.216）**：**把全局加载组织成 128 字节对齐的合并（coalesced）段**，干净映射到缓存线——避免事务拆分，最大化 L2 与 DRAM 带宽利用。

**（6）全局显存（HBM，Figure 6-14）**：溢出的寄存器/超大自动数组（local memory）也落在这里，尽管 HBM3e 有 ~8 TB/s，**高延迟使它成为链条上最慢的一环**。

**工具**：用 Nsight Compute 追踪 **spill 和缓存命中率**，让 kernel 尽可能贴着片上峰值运行。

📦 **边栏 Note（p.217，冷知识）**：**B200 以单 GPU 呈现（统一全局地址空间），物理上是两块 reticle-limited（受光刻极限限制的）die**，由 **10 TB/s 片间互连**相连；每块 die 接 4 个 HBM3e 栈（共 8 个）。开发者视角访存是均匀的，但值得了解底层。

**Point of Coherency（内存一致性生效点，Figure 6-15）**：一致性按线程通信的层级发生在不同位置——**thread / thread block（CTA）/ thread block cluster（CTA cluster）/ device / system** 五级。通信范围越广，一致性代价越高。

## §2.6 Unified Memory（统一内存，p.218–221）

**动机**：Grace Hopper / Grace Blackwell 这类 **CPU-GPU 统一超级芯片**设计下，理解统一内存尤为重要。

**机制（Figure 6-16）**：Unified Memory（又名 CUDA Managed Memory）给你**单一、一致的 CPU+GPU 地址空间**——不再需要分别维护 host/device 缓冲、不再手写 cudaMemcpy。底层：CUDA 运行时给每个 `cudaMallocManaged()` 分配的内存以**页**为单位背书，页可以**按需（on-demand）**跨 CPU-GPU 互连迁移。

**代价**：
- 超级开发者友好，但会引发**不想要的按需页迁移** → 隐藏延迟和执行停顿；
- GPU 线程访问一个当前在 CPU 内存的数据 → **GPU 缺页（page fault）并等待**该数据经互连传来；
- 性能强依赖硬件：传统 PCIe / 早期 NVLink 上迁移带宽低——**按缺页搬运常常比手动 cudaMemcpy 还慢**；Grace 超级芯片的 **NVLink-C2C 提供最高 ~900 GB/s**（CPU 内存 ↔ GPU HBM3e），缺页迁移接近设备原生速度——**但延迟仍非零**；
- **kernel 执行中的意外缺页会让 GPU 停等**运行时把页挪到位。

**治理手段**（逐一对应插图）：

1. **预取（Figure 6-17）**：`cudaMemPrefetchAsync()` 提示驱动**在 kernel 启动前**把指定范围搬到目标 GPU（或 CPU）——把昂贵的"首次触碰迁移"变成**可重叠的异步传输**。
2. **内存建议（Figure 6-18）**：
   ```cuda
   cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, gpuId); // 数据主要在哪用
   cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, gpuId);        // 基本只读
   cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, otherGpuId);   // 让另一块 GPU 直接映射、不触发迁移
   ```
3. **绑定 stream**：默认**任何** stream/kernel 都可能对 managed 内存触发缺页 → 意外迁移 + 隐式同步。若确定某缓冲区一段时间内只被一条 stream/一块 GPU 用：
   ```cuda
   cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachSingle);
   ```
   之后**只有该 stream 的操作**会对它缺页迁移——迁移与其他 stream 的工作重叠、防止别的 stream 意外被它卡住、避免跨 stream 同步。

📦 **边栏 Note（p.221）**：**没有 NVLink-C2C 的多 GPU 系统**里，还可以用 `cudaMemcpyPeerAsync()` 或预取到特定设备，把数据钉在 **NUMA 本地**的 GPU 显存里，防止慢速远程访问。

**小结**：
> 原句："With techniques like proactive prefetching, targeted memory advice, and stream attachment, Unified Memory can deliver performance very close to manual cudaMemcpy while preserving the simplicity of a unified address space."
> 主动预取 + 定向内存建议 + stream 绑定，统一内存能在保住"统一地址空间"简洁性的同时，达到非常接近手动 cudaMemcpy 的性能。数据在 kernel 跑之前就已经在它该在的地方。

## §2.7 Maintaining High Occupancy and GPU Utilization（维持高占用率与 GPU 利用率，p.221–229）⭐核心案例

**原理重申**：GPU 靠同时跑很多 warp 维持性能——一个 warp 等数据时另一个能跑，快速 warp 切换即**隐藏内存延迟**。SM 容量中被活跃 warp 实际占据的比例 = **occupancy**。

- occupancy 低（活跃 warp 少）→ 一个 warp 等内存时 SM 干瞪眼 → SM 利用率差；
- Blackwell 的**大寄存器堆（64K/SM）**让高 occupancy 更容易——能养更多 warp 而不溢出；
- 📦 **边栏 Tip**：每线程最多 255 寄存器；**用 profiler 检查 achieved occupancy（实际占用率），据此调 block size 和寄存器用量**。
- occupancy 高 → 一个 warp 等访存时其他 warp 换入执行 → 掩盖长访存延迟 = **hiding latency**。

**本章最基本的 CUDA 性能优化法则**：
> 原句：**"launch enough parallel work to fully occupy the GPU."**
> 启动足够多的并行工作，把 GPU 填满。

**两条判断规则**（p.222，重要）：
1. 若 achieved occupancy 远低于上限**且**性能差 → 第一味药是**加并行度**（更多 block/线程），把 occupancy 推向现代 GPU 上的 **80%–100%** 区间；
2. 若 occupancy 已中高**但 kernel 受限于内存吞吐** → 推到 100% 也没用。**你只需要"刚好够藏延迟"的 warp 数**，超过之后瓶颈在别处（如显存带宽）。

### 案例：向量加法的两种实现

**实现一：addSequential（反面教材）**——只让 `blockIdx.x==0 && threadIdx.x==0` 的一个线程 for 循环加完全部 N=1M 元素；启动配置 `<<<1,1>>>`（书中注释特意强调：**别改这个启动配置**，这个 kernel 的语义就是单线程）。
> 原句："In this single-threaded version, the GPU's vast resources are mostly idle. Only one warp, or even one thread within the warp, is doing work while all others sit idle. The result is very poor occupancy and, ultimately, low performance."
> 单线程版本里 GPU 的庞大资源基本闲置——只有一个 warp、甚至只有其中一个线程在干活。结果：极差的 occupancy、极低的性能。

**PyTorch 的同款错误**（p.223，作者特意用一节警告）：
```python
# 反面教材：Python for 循环逐元素加 —— DO NOT DO THIS
with torch.inference_mode():
    for i in range(N):        # 串行发射 N 个迷你 GPU 操作
        C[i] = A[i] + B[i]
```
这等于**把 GPU 当标量、非并行处理器用**，occupancy 与 addSequential 一样惨。间接执行高层框架里的低效 GPU 代码，一样要警惕。

**实现二：addParallel（正确姿势，Figure 6-19 向量化加法）**——`<<<(N+255)/256, 256>>>` 每线程加一个元素。书中的完整版还示范了全套好习惯：`__restrict__` 指针注解（帮编译器确认无别名）、pinned host 内存、非阻塞 stream、`cudaMallocAsync`/`cudaMemcpyAsync`/`cudaFreeAsync` 全异步链路。
PyTorch 正确姿势就一行：**`C = A + B`**（单个向量化 kernel 并行处理所有元素）。

📦 **边栏 Note（p.226）**：实践中 PyTorch 的向量化张量操作会"做对的事"。**要小心的是在 GPU 操作外面套 Python 级循环**——那会串行化工作、拖垮性能。除非你在写全新的东西，几乎总有优化好的 PyTorch 原生实现（包括 PyTorch 编译器生成的代码）。

### 量化验证：nsys + ncu

书中给出完整命令行（值得收藏）：
```bash
# Nsight Systems：整体时间线
nsys profile --stats=true -t cuda,nvtx -o report ./app.py
# Nsight Compute：单 kernel 指标
ncu --section SpeedOfLight \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.avg \
    --target-processes all --print-summary per-gpu -o report ./app.py
```
分工：**nsys 揭示时间花在哪、GPU 是被"饿着"还是被"堵着"；ncu 解释 kernel 为什么是这个表现**（比如 occupancy 差）。只跑 nsys 会漏掉 kernel 内的细粒度低效；只跑 ncu 不知道 kernel 有没有被及时喂数据。

**Table 6-6：串行 vs 并行 kernel 对比**（示意值）

| 指标 | add_sequential | add_parallel |
|---|---|---|
| Kernel 执行时间 (ms) | 48.21 | **2.17** |
| GPU 利用率 | 1.5% | **95%** |
| Achieved occupancy | 1.3% | **38.7%** |
| Warp 执行效率 | 3.1% | **100%** |

📦 **边栏 Note**：不同工具对指标叫法不同——Nsight Systems 报整体 "GPU Utilization"，Nsight Compute 给每 kernel 的 "SM Active %"，**都反映 SM 被活跃 warp 占据的充分程度**。

**解读**（p.227）：
- 从单线程单 warp → 全并行多 warp：occupancy 1.3% → ~38.7%，运行时间 **约 22×**（48.21 → 2.17 ms）；
- 串行版只有一个 SM 上的一个线程干活 → GPU 利用率 1.5%；并行版多 SM 多活跃 warp → 95%；
- warp 执行效率 3.1% → 100%：并行版每条指令期间 **warp 里 32 个线程全在干有用的活**（串行版 1/32 ≈ 3.1%）；
- **warp 级延迟隐藏**的图景（Figure 6-20 时间线对比）：一个 warp 等内存加载时，另一个在做加法、又一个在取下一批数据——串行版没有"别的 warp"可切，硬件管线只能空转，时间线是"一长串操作 + 访存等待的空档"；并行版**空档被其他 warp 的工作填满**，GPU 持续忙碌。

**关键递进 + 冷水**（p.228）：
1. 第一步永远是**保证足够的并行工作**填满 GPU；
2. 线程够了之后，下一步才是优化每个 warp 的执行效率（ILP 等每线程改进）；
3. **但即使 100% occupancy，若工作负载是 memory bound（受限于慢速访存而非计算），性能照样受损。**

**Memory-bound 的现实例子——LLM decode**（p.228–229）：
- decode（逐 token 生成）阶段要把**模型权重**从全局 HBM 搬进寄存器和共享内存；
- 现代 LLM 有几千亿参数（按 8 bit/参数算就是几百 GB）——这个搬运量轻松打满显存带宽；
- 📦 **边栏 Note**：**GPU FLOPS 的增速正在甩开显存带宽**——Blackwell HBM3e ~8 TB/s，但算力和模型规模长得更快。**优化数据搬运对现代 AI 负载绝对关键。**

## §2.8 Tuning Occupancy with Launch Bounds（用 Launch Bounds 调占用率，p.229–231）

**动机**：有时"多开线程"不够——尤其每线程资源（寄存器/共享内存）消耗大时。可以用 **`__launch_bounds__`** 注解在**编译期**引导编译器为 occupancy 优化。

```cuda
__global__ __launch_bounds__(256, 16)
void myKernel(...) { /* ... */ }
```

语义：**承诺**该 kernel 启动时每 block 绝不超过 256 线程；**请求**编译器把寄存器分配和内联控制到"每 SM 至少能驻 16 个 256 线程的 block（= 4096 线程）"。这些提示影响编译器的**寄存器分配和内联决策**。

📦 **边栏 Note**：记住硬件上限——每 block 1024 线程、每 SM 最多 2048 常驻线程。

**硬件上限的裁剪**：4096 超过了 SM 的 2048 → 编译器把请求**削到硬件最大**（8 block × 256 线程），并发出警告，形如：
> "ptxas warning: Value of threads per SM…is out of range. .minnctapersm will be ignored."

**实际效果与代价**（p.230）：`__launch_bounds__` 常使编译器**限制每线程寄存器用量**（有时还限制展开/内联）来避免溢出、允许更高 occupancy——本质是**牺牲一点单线程性能**（不用满每个寄存器、不展开到极限），**换取更多 warp 在飞、更稳定的 warp 吞吐**。

⚠️ 增加 occupancy 必须与每线程资源平衡：**强行塞太多线程 → 寄存器不够 → 溢出到 local memory → 慢访存**。

**运行时自动调优：CUDA Occupancy API**

```cuda
int minGridSize = 0, bestBlockSize = 0;
size_t dynSmemBytes = /* 若用 extern __shared__，如实填每 block 字节数 */ 0;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize, &bestBlockSize, myKernel,
    dynSmemBytes, /* blockSizeLimit = */ 0);
// 覆盖 N 个元素，但不低于"喂饱 occupancy"的最小 grid：
int gridSize = std::max(minGridSize, (N + bestBlockSize - 1) / bestBlockSize);
myKernel<<<gridSize, bestBlockSize, dynSmemBytes>>>(...);
```

- API 依据 kernel 的**实际资源用量**（寄存器、共享内存）计算最可能优化 occupancy 的 block size；
- **两个易错点**（书里专门强调）：① `minGridSize` 是"**让 occupancy 饱和的最小 grid**"，**不是**覆盖长度 N 的正确 grid——要取 `max(minGridSize, ceil_div(N, bestBlockSize))`；② kernel 若用 `extern __shared__` 动态共享内存，**必须如实传入字节数**。

📦 **边栏 Tip（p.231）**：**用 ±1–2 档候选 block size 实测验证** Occupancy API 的建议——现代 GPU 上寄存器压力和 L2 行为可能让**略低于最大 occupancy 的配置实际更快**。

📦 **边栏 Note（p.231，寄存器-occupancy 权衡总纲）**：
> 每线程用更少寄存器（或用 launch bounds 封顶）→ 更多 warp 常驻 → 延迟隐藏更好。但寄存器太少 → 编译器被迫溢出数据到 local memory → 性能受损。**找甜点常需实验**。Nsight Compute 的 "Registers Per Thread" 和 "Occupancy" 指标是向导。

**使用时机**：编译器启发式通常不错；当你**确知** kernel 可以拿每线程资源换更多活跃 warp 时，`__launch_bounds__` 和 occupancy 计算器给你显式控制，防止"重线程"导致 SM 欠占用。

---

# §3 Debugging Functional Correctness with NVIDIA Compute Sanitizer（p.231–232）

**动机**：CUDA 应用每个 kernel 派生几千线程，传统调试抓不住细微的内存 bug 和竞争条件。**Compute Sanitizer**（CUDA Toolkit 自带的功能正确性套件）在**运行时插桩**代码、在开发早期发现错误——减少调试往返、提高代码可靠性。

**CLI 用法**：
```bash
compute-sanitizer [--tool 工具名] [选项] <应用> [应用参数]
```
- 支持 **NVTX** 注解做细粒度分析（书中说 NVTX 应该在正确性和性能分析中**广泛使用**）；
- `--error-exitcode`：出错时以非零码退出（CI 关键）；
- `--kernel-name` / `--kernel-name-exclude`：只对特定 kernel 消毒；
- `--nvtx yes`：启用 NVTX，缩小分析范围、减少内存泄漏报告的误报。

📦 **边栏 Tip**：**把 Compute Sanitizer 集成进 CI 流水线**（配 `--error-exitcode` + kernel 过滤 + NVTX 区域注解），拦截正确性回归。

**四大工具**：

| 工具 | 检测什么 | 细节 |
|---|---|---|
| **memcheck** | 越界/未对齐访问；GPU 硬件异常；device 侧内存泄漏 | 精确定位并归因于 global/local/shared 内存；`--check-device-heap` 附加检查堆分配 |
| **racecheck** | 共享内存数据冒险 | **Write-After-Write（写后写）、Write-After-Read（读后写）、Read-After-Write（写后读）** 三类，可导致非确定性行为；验证 warp 内与 block 内的线程间通信正确性 |
| **initcheck** | 访问**未初始化的** device 全局内存 | 病因常是缺 H2D 拷贝或跳过了 device 侧写入；防"陈旧/垃圾数据"型隐蔽 bug |
| **synccheck** | 无效同步原语 | 如错配的 barrier；识别可导致死锁和跨线程状态不一致的线程顺序冒险 |

**小结**：四件套 + CI 集成 = 内存、竞争、初始化、同步四类 bug 的早期发现——"ship reliable, high-performance code with confidence"。

---

# §4 Roofline Model：Compute-Bound 还是 Memory-Bound（p.233–236）

## §4.1 模型定义（p.233）

Roofline 是一种可视化，画出**两条硬件性能天花板**：
- **水平线**：处理器峰值浮点速率（compute roof，计算屋顶）；
- **斜线**：峰值显存带宽决定的上限（memory roof，内存屋顶）。

两线合成一个"屋顶"包络，揭示 kernel 受限于**计算**（compute bound）还是**数据搬运**（memory bound）。

- 交点 = **ridge point（脊点）**：kernel 从 memory bound（脊点左侧）过渡到 compute bound（右侧）的**算术强度阈值**；
- **Arithmetic intensity（算术强度）**：每与片外全局显存交换 1 字节所做的 FLOPs 数（FLOPs/byte）。

## §4.2 算例（p.233–234，Figure 6-21）

书中的示例 kernel：加载两个 32 位 float（8 字节）、加一次（1 FLOP）、写回一个 32 位 float（4 字节）：
- 算术强度 = 1 FLOP ÷ 12 字节 ≈ **0.083 FLOPs/byte**；
- Blackwell 的脊点 ≈ **10 FLOPs/byte**（~80 TFLOPs ÷ 8 TB/s）；
- 0.083 比 10 低**两个数量级以上（>100×）**——完全喂不饱 ALU（arithmetic logic unit，算术逻辑单元），钉死在斜线（内存带宽天花板）上，**性能由访存停顿而非计算主导**。

**优化方向**：提高算术强度=每字节多做工作 → 点在图上**右移** → 性能向计算屋顶爬升。

## §4.3 降低精度：最直接的右移手段（p.234–235）

- FP32 → **FP16**：每操作传输字节减半 → **FLOPs/byte 立刻翻倍**；
- 现代 GPU 有专用 **FP8 Tensor Core**；Blackwell 新增原生 **FP4 Tensor Core**（面向部分 AI 负载，如推理）——字节/操作进一步下降、强度进一步上升；
- Blackwell 的 FP8 相对 FP16：**吞吐翻倍、内存占用减半**；
- **一条 128 字节访存事务的装载量：32 个 FP32 = 64 个 FP16 = 128 个 FP8 = 256 个 FP4**；
- **硬件解压（hardware decompression）**：模型可以**压缩形式**存放于 HBM（甚至超出 FP4 的压缩，如 4 位/2 位方案），硬件在读取时**在线解压**、再 cast 到 FP16/FP32 做高精度聚合计算——**实质提高了读权重时的可用显存带宽**；
- 因此 **Blackwell 对 memory-bound 负载（如 transformer 的 token 生成）有架构级优势**。

📦 **边栏 Note（p.235，重要的框架视角）**：
> "Transformer-based models (e.g., LLMs) can be both compute bound and memory bound in different phases. For example, attention layers (prefill phase) are typically compute bound, while matrix multiplications (decode phase) are often memory bound."
> Transformer 模型在不同阶段既可能 compute bound 也可能 memory bound：**attention 层（prefill 阶段）通常算力受限，矩阵乘（decode 阶段）常常访存受限**。第 15–18 章深入推理时详谈。

## §4.4 用 Nsight 诊断与验证（p.235–236）

**memory-bound 的指标签名**：Nsight Compute 报告**很高的 DRAM 带宽利用率 + 很低的实际计算指标（如低 ALU 利用率）** → warp 大部分时间停在访存上。

**工具组合**：
- **Nsight Compute**：每 kernel 计数器——延迟、缓存命中率、warp 发射停顿（warp issue stalls）；新版本还有 **range replay（区间重放，带指令级源码指标）**、改进的**源码关联导航**、**launch-stack size 指标**——更快诊断依赖停顿、寄存器压力、启动配置的影响；
- **Nsight Systems**：整体时间线——GPU 空闲缺口、与 CPU 工作的重叠、PCIe/NVLink 传输；
- 合起来：**ncu 给"为什么"（哪些停顿、哪些资源），nsys 给"什么时候"（停顿嵌在整体执行的哪里）**。

**迭代方法**：反复 profile、用两个工具的指标定位内存热点；给可疑代码加 **NVTX range**、放大时间线行为、按反馈优化。例：用 NVTX 标注"memory copy"和"kernel execution"区域，在 nsys 时间线确认 host-device 传输与计算的重叠（Figure 6-22：同步/顺序 vs 异步/重叠的数据传输对比）。

**重叠失败的两大病因**（p.236）：
1. 预期重叠却看到拷贝和 kernel 串行 → 多半是**不想要的默认流同步**；
2. 或者**缺 pinned-memory 缓冲**。
📦 **边栏 Warning**：**不用 pinned（页锁定）内存，cudaMemcpyAsync 无法与 kernel 执行重叠——这是常见性能问题。**

**"kernel 被饿着"的信号与修复后的样子**：ncu 里 **global load efficiency（全局加载效率）下降** = DRAM 请求满足得不够快；nsys 时间线上 kernel 启动之间出现**空闲段**（GPU 在等数据）。应用本章内存层级优化后：空闲段基本消失、ncu 的 memory pipe utilization 爬向峰值、端到端吞吐跳升。

📦 **边栏 Tip（p.236）**：**每次改动后都要测量**——profiling 工具会确认优化是否真的减少了访存停顿。

---

# §5 Key Takeaways（要点，p.237–238，书中 8 条完整版）

1. **SIMT 执行模型**：GPU 以 32 线程 warp 锁步发射指令；高 occupancy（多 warp 在飞）隐藏内存与管线延迟。
2. **线程层级 threads → blocks → grids**：线程组成 block（≤1024），block 组成 grid，扩展到百万线程不改代码。同步（`__syncthreads()` 或 cooperative groups）支持共享内存数据复用但有开销——**尽量少设 barrier**。
3. **Occupancy vs 资源上限**：block size 取 32 的倍数避免欠填充 warp、最大化调度器利用。牢记 per-SM 上限：Blackwell **每线程 255 寄存器、每 SM 228 KB 共享内存、64 常驻 warp、32 常驻 block**。
4. **Kernel 启动参数**：从 `threadsPerBlock=256`（8 warp）起步平衡 occupancy 与资源；`blocksPerGrid=(N+tpb−1)/tpb` 覆盖所有元素；根据 profiling（寄存器/溢出、共享内存用量、achieved occupancy）调整。
5. **异步内存管理**：优先 `cudaMallocAsync`/`cudaFreeAsync` + 专用 stream + CUDA 内存池，避免全局同步和 OS 开销。PyTorch 的 caching allocator 是同一思路（避免昂贵的 cudaMalloc/cudaFree 调用）。
6. **GPU 内存层级**：寄存器 → L1/共享 → L2 → 全局（HBM3e）→ host：每级以容量换延迟/带宽。**把数据复用最大化地放在寄存器和共享/L1。**
7. **Unified Memory 注意事项**：简化编程但可能引发隐式页迁移；用 `cudaMemPrefetchAsync` + 内存建议避免"惊吓式"停顿。
8. **Roofline 分析**：算术强度（FLOPs/byte）决定 memory bound 还是 compute bound。用低精度（FP16/FP8/FP4 + 硬件解压）提升 FLOPs/byte、把 kernel 推向计算屋顶。用 Nsight Compute（每 kernel 指标）+ Nsight Systems（时间线）定位并消除访存停顿。**TMEM + Blackwell UMMA 结合 FP8/FP4，能把 kernel 从 memory-bound 拉向 compute-bound**（UMMA 第 10 章细讲）。

---

# §6 Conclusion（结论，p.238）

本章为高性能 CUDA 开发打了地基：SIMT 模型、线程层级、多级内存系统。**Occupancy（活跃 warp 与理论上限之比）对延迟隐藏很重要。**

**但全章最重要的转折在结论里**：
> 原句：**"However, maximizing occupancy does not guarantee best performance in every case. GPUs can often achieve very high throughput at moderate or even low occupancy if threads have sufficient instruction-level parallelism (ILP)—or if other resources are the bottleneck."**
> 占用率最大化**不保证**每种情况的最优性能。若线程有足够的**指令级并行（ILP）**——或瓶颈在其他资源——GPU 常能在中等甚至较低 occupancy 下打出很高吞吐。

> 原句：**"there are scenarios in which reducing the number of active threads will free up registers for other threads. This allows more computations per thread—and ultimately boosts throughput. Always benchmark different occupancy levels to find the optimal setting for your workload and hardware."**
> 某些场景下**减少活跃线程数**反而释放寄存器给其余线程、让每线程做更多计算、最终提升吞吐。**永远对不同 occupancy 水平做 benchmark**，为你的负载和硬件找最优设置。

**下一步预告**：带着这些基本功和 profiling 技巧，接下来进入针对性优化——避免 warp 分叉、利用 GPU 内存层级、异步预取内存；还会深入 **TMA**（处理批量内存搬运、解放 GPU 专注于有用工作、提高计算 goodput）。

---

# 附：本章插图索引（22 张，图号 → 内容一句话）

| 图 | 内容 |
|---|---|
| 6-1 | 简单 CUDA 编程流程（CPU↔GPU 数据搬运 + kernel 调用） |
| 6-2 | Blackwell SM 的 4 个 warp 调度器与双发射 |
| 6-3 | Threads / thread blocks（CTA）/ grids 三层 |
| 6-4 | 线程层级 + host 发起 kernel 的视角 |
| 6-5 | Thread block cluster 的硬件 DSMEM |
| 6-6 | block 内两段代码之间的 `__syncthreads()` |
| 6-7 | Warp（32 线程）在 warp 调度器管理下整体推进 |
| 6-8 | SIMT warp 分叉（左）vs 一致（右） |
| 6-9 | Blackwell 上线程资源的相对规模与上限 |
| 6-10 | GPU 内存层级（含 CPU） |
| 6-11 | TMEM+SMEM 服务 Tensor Core 做 C=A×B |
| 6-12 | 每线程的 local memory |
| 6-13 | Thread block 共享内存 |
| 6-14 | 每设备全局显存（HBM） |
| 6-15 | 内存一致性生效点（thread→system 五级） |
| 6-16 | Unified Memory 自动页迁移 |
| 6-17 | cudaMemPrefetchAsync 经 NVLink-C2C 流式预取 |
| 6-18 | cudaMemAdvise 指定 preferred location / ReadMostly |
| 6-19 | 向量化加法的并行执行 |
| 6-20 | 并行 vs 串行时间线对比（空档被其他 warp 填满） |
| 6-21 | Blackwell 级 roofline（~80 TFLOPs、8 TB/s、脊点 10、示例点 0.083） |
| 6-22 | 同步（顺序）vs 异步（重叠）传输+计算 |
