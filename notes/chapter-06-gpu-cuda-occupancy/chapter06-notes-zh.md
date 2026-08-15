# 第 6 章读书笔记：GPU 架构、CUDA 编程与最大化占用率（Occupancy）

> 对应原书 Chapter 6（PDF 第 219–266 页），配套讲解 PPT：`chapter06-presentation.pdf`

## 一句话总结

GPU 是"吞吐量机器"：靠成千上万的线程同时干活来掩盖内存延迟。本章讲三件事——**线程组织方式**（warp/block/grid）、**内存层级**（寄存器 → 共享内存 → L2 → HBM）、以及**第一条黄金法则：先把 GPU 喂饱**（launch 足够多的并行工作，即高 occupancy）。

---

## 1. GPU 和 CPU 的本质区别

- **CPU 优化延迟**：核少、缓存深，追求单个线程跑得快。
- **GPU 优化吞吐**：几百个 SM（流式多处理器），同时跑几千个线程。
- 大白话：**CPU 是跑车，GPU 是货运列车**——别指望列车快，要让列车"装满"。
- 基本 CUDA 流程：CPU 把数据拷到 GPU 显存 → 启动 kernel → 把结果拷回来。GPU 靠海量并行把这些传输延迟"藏"起来。

## 2. 线程层级：Thread → Block → Grid

| 概念 | 大白话 | 关键数字（Blackwell B200） |
|---|---|---|
| Thread（线程） | 一个工人，处理一个数据元素 | — |
| Warp | 32 个线程绑在一起、同步执行的"划船队" | 固定 32 |
| Thread Block（CTA） | 一个"班组"，组内可用共享内存快速交流、可同步 | 最多 1,024 线程（= 32 个 warp） |
| Grid | 一次 kernel 启动的全部班组 | X 维最多约 21 亿个 block |

- **SIMT**（单指令多线程）：一个 warp 里 32 个线程步调一致地执行同一条指令。大白话：32 人划船队，必须同一拍划桨。
- **Warp divergence（warp 分叉）**：同一个 warp 内如果有人走 `if`、有人走 `else`，硬件只能先执行一条路径（把另一半"蒙住"），再执行另一条——时间按分支数翻倍。**不同 warp 之间走不同分支没有任何代价**。
- Block 之间**执行顺序不保证、互相独立**——正是这个约束让同一份代码在未来更多 SM 的 GPU 上自动扩展。
- 新硬件特性：**Thread Block Cluster + DSMEM**（跨 SM 的分布式共享内存），让多个 block 之间也能共享内存（第 10 章细讲）。

## 3. SM 内部结构（Blackwell）

- 每个 SM 相当于 **4 个"迷你 SM"**：4 个独立 warp 调度器，共享片上资源。
- 每个调度器每拍发射 1 个 warp 的指令，且支持**双发射（dual-issue）**：同一 warp 同一拍可以发 1 条算术指令 + 1 条访存指令。
- 最好情况：每拍 4 条数学 + 4 条访存指令同时飞。
- **SFU**（特殊功能单元，算 sin/cos/sqrt 等）走独立管线，不占用主管线。
- 每 SM 资源：**64K 个 32 位寄存器**（256 KB，单线程上限 255 个）、**256 KB 统一 L1/共享内存**（其中最多 228 KB 可作共享内存，实际可用 227 KB）。
- 每 SM 上限：**64 个常驻 warp（= 2,048 线程）、32 个常驻 block**。

## 4. CUDA 编程速览

### Kernel 的骨架（必背）

```cuda
__global__ void myKernel(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 全局唯一索引
    if (idx < N) {          // 边界检查，防止越界
        input[idx] *= 2.0f;
    }
}
// 主机侧：
int threadsPerBlock = 256;                                  // 32 的倍数
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // 向上取整
myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
```

- 大白话：**你只写"一个工人的作业说明"，CUDA 复印一百万份，每份发一个不同的行号（idx）**。
- `if (idx < N)` 边界检查：最后一个 block 通常填不满，多出来的线程直接退出，否则会报 `cudaErrorIllegalAddress`。
- CUDA 错误是**懒惰上报**的（kernel 异步执行）：要在启动后用 `cudaGetLastError()` + `cudaDeviceSynchronize()` 主动查错。

### 启动参数怎么选

- **threadsPerBlock 从 256 开始**（8 个 warp），是"中杯咖啡"——几乎不会点错，之后按 profiling 调 128/512。
- 必须是 **32 的倍数**：33 线程的 block 要占两个 warp 槽位，第二个 warp 只有 1/32 在干活。
- 2D/3D 数据用 `dim3`（如 16×16 的 block 处理图像），套路完全一样。

### 异步内存分配（重要习惯）

- `cudaMalloc`/`cudaFree` 是**同步且贵**的：全设备同步 + 操作系统调用。
- 推荐 `cudaMallocAsync`/`cudaFreeAsync` + 非阻塞 stream（`cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`），底层用**内存池**复用已释放的块。
- 大白话：**别每次送货都买车再卖车（手续费很贵），在停车场养一支车队，用时拿钥匙就走**。
- PyTorch 的 caching allocator（`PYTORCH_ALLOC_CONF`）就是同样的思路。

## 5. GPU 内存层级（Blackwell 数字）

| 层级 | 容量 | 延迟 | 带宽 |
|---|---|---|---|
| 寄存器（每线程） | 256 KB/SM，255/线程 | ~1 拍 | 几十 TB/s |
| 共享内存 + L1（每 SM） | 228 KB 可配 | 20–30 拍 | TB/s 级 |
| TMEM（每 SM，Tensor Core 专用） | 256 KB | ~10 拍 | TB/s 级 |
| 常量缓存 | 8 KB（前置 64 KB 常量区） | 1 拍广播 | 全 warp 同地址读=寄存器速度 |
| L2（全 GPU 共享） | **126 MB** | ~200 拍 | 数十 TB/s |
| Local memory（寄存器溢出区） | 落在 DRAM | 数百~1000 拍 | 同 HBM |
| 全局显存 HBM3e | 180 GB（B200）/288 GB（B300） | 数百~1000 拍 | **~8 TB/s** |

- 大白话：**书桌（寄存器）→ 办公室书架（共享内存）→ 楼里图书馆（L2）→ 城另一头的仓库（HBM）。尽量在书桌上干活。**
- 访问 HBM 时要 **coalesced（合并访存）**：128 字节对齐的连续访问，正好映射一条缓存线。
- **TMEM/TMA**：TMEM 是 Tensor Core 的累加器内存（不能用指针直接访问）；TMA（张量内存加速器）像"专职叉车队"，用描述符在 HBM↔SMEM↔TMEM 之间自动搬运数据，让 Tensor Core 不停工（第 10 章细讲）。
- B200 实际是**两块 die 拼的**（10 TB/s 片间互连、共 8 个 HBM3e 栈），但对开发者呈现为一个统一地址空间。

## 6. Unified Memory（统一内存）

- `cudaMallocManaged()`：CPU/GPU 一个地址空间，页**按需自动迁移**，不用手写 `cudaMemcpy`。
- 代价：GPU 碰到还在 CPU 侧的页会**缺页中断并停顿**。Grace 超级芯片上走 NVLink-C2C（~900 GB/s）迁移便宜很多，但不是免费。
- 大白话：**统一内存是客房服务——方便，但不提前点单（prefetch），开饭时就得饿着等**。
- 三板斧消除"意外停顿"：
  - `cudaMemPrefetchAsync(ptr, size, gpuId, stream)`——启动 kernel 前先把数据搬过去；
  - `cudaMemAdvise(...SetPreferredLocation / SetReadMostly / SetAccessedBy...)`——告诉驱动数据主要在哪用、是否只读；
  - `cudaStreamAttachMemAsync(...)`——把一段内存绑到一个 stream，避免跨 stream 的意外同步。

## 7. Occupancy（占用率）：本章核心

- **定义**：SM 上活跃 warp 数 / 硬件上限（64）。高 occupancy = 一个 warp 等内存时总有别的 warp 顶上 = **延迟被藏住**。
- **反面教材**：单线程 kernel（`<<<1,1>>>` 里 for 循环跑完全部数据），或 PyTorch 里用 Python for 循环逐元素 `C[i]=A[i]+B[i]`（发射 N 个串行小 kernel）。这等于**把 GPU 当很贵的单核 CPU 用**。
- 实验对比（书中示意数字，Table 6-6）：

| 指标 | 串行版 | 并行版 |
|---|---|---|
| Kernel 时间 | 48.21 ms | **2.17 ms（22×）** |
| GPU 利用率 | 1.5% | 95% |
| Achieved occupancy | 1.3% | 38.7% |
| Warp 执行效率 | 3.1% | 100% |

- **注意**：高 occupancy 是必要非充分。如果 kernel 是 memory-bound，占用率拉满也没用——瓶颈在带宽（见 roofline）。

### 调 Occupancy 的工具

- `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)`：向编译器承诺 block 上限，让它控制寄存器用量、多塞几个 warp。超硬件上限的请求会被削到 2,048 线程/SM 并警告。
- `cudaOccupancyMaxPotentialBlockSize()`：运行时自动算最优 block size。注意返回的 `minGridSize` 是"喂饱 occupancy 的最小 grid"，实际 grid 要取 `max(minGridSize, ceil(N/blockSize))`。
- **权衡**：寄存器少 → warp 多 → 延迟藏得好；但太少 → 寄存器溢出到 local memory（很慢）。**用 Nsight Compute 的 Registers Per Thread + Occupancy 指标找甜点，并实测 ±1–2 档 block size**。

## 8. Compute Sanitizer（功能正确性）

四件套（`compute-sanitizer --tool <名字> ./app`）：

- **memcheck**：越界/未对齐访存、显存泄漏；
- **racecheck**：共享内存数据竞争（WAW/WAR/RAW）；
- **initcheck**：读了未初始化的全局显存（常见原因：忘了 H2D 拷贝）；
- **synccheck**：非法同步/错配 barrier。

建议进 CI（`--error-exitcode`）。大白话：十万个线程的程序，"我这跑过没问题"毫无意义——sanitizer 是竞争条件的安全带。

## 9. Roofline 模型（屋顶线）

- **算术强度（arithmetic intensity）= FLOPs / 搬运字节数**。
- 两条"天花板"：水平线 = 峰值算力（~80 TFLOP/s FP32），斜线 = 峰值带宽（~8 TB/s）。交点叫 **ridge point ≈ 10 FLOPs/byte**。
- 在 ridge 左边 = **memory-bound**（喂不饱 ALU），右边 = **compute-bound**。
- 例子：`C = A + B` 每 12 字节只做 1 次浮点加 → 0.083 FLOPs/byte，比 ridge 低 100 多倍，铁定 memory-bound。
- **往右移的办法 = 每字节多干活**：
  - 片上复用数据、算子融合；
  - **降低精度**：一条 128 字节访存事务能装 32 个 FP32 = 64 个 FP16 = 128 个 FP8 = 256 个 FP4。FP32→FP16 直接让算术强度翻倍；
  - Blackwell 的**硬件解压**：权重压缩存在 HBM 里，读取时硬件现场解压——变相增加可用带宽。这就是它跑 memory-bound 的 LLM decode（逐 token 生成）特别强的原因。
- LLM 两阶段的直觉：**prefill 偏 compute-bound，decode 偏 memory-bound**（要把几百 GB 权重反复从 HBM 搬进片上）。

## 10. Profiling 方法论

- **Nsight Systems（nsys）= 整机秒表**：看时间花在哪、GPU 是否在空转、传输和计算有没有重叠。
- **Nsight Compute（ncu）= 单 kernel 显微镜**：看为什么慢——occupancy、stall 原因、缓存命中率、DRAM 利用率。
- 两个都要用：只用 nsys 看不到 kernel 内部低效；只用 ncu 不知道 kernel 是不是"没被喂饱"。
- 用 **NVTX** 给可疑代码段打标签，在时间线上直接看到。
- 常见坑：`cudaMemcpyAsync` 想跟计算重叠，**必须用 pinned（页锁定）内存**，否则重叠不了；另一个常见原因是默认 stream 的隐式同步。
- **每改一处就测一次**——profiler 会告诉你优化是否真的减少了 stall。

## 11. 容易被问到的点（答辩预备）

1. **为什么 kernel 要传 N？** kernel 是"单线程视角"的函数，N 定义了本次要处理的数据边界，配合 idx 做边界检查。
2. **Occupancy 是不是越高越好？** 不是。足够藏延迟即可；memory-bound 时再高也没用；有时降低线程数、给每线程更多寄存器（更多 ILP）反而更快。永远以实测为准。
3. **PTX 和前向兼容**：只编译特定架构的 SASS（如 sm_90）不能在新架构上跑；要在 fatbin 里带 PTX（可用 `CUDA_FORCE_PTX_JIT=1` 验证）。
4. **共享内存 228 KB 但 block 只能申请 227 KB**：CUDA 保留 1 KB。
