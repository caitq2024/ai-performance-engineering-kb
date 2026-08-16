# 第 12 章读书笔记：动态调度、CUDA Graphs 与设备侧 Kernel 编排

> 对应原书 Chapter 12（PDF 第 509–554 页），配套讲解 PPT：`chapter12-presentation.pdf`

## 一句话总结

第 6–11 章解决"单个 kernel 怎么快"；本章解决"kernel 之间别让 GPU 闲着"。核心思想只有一个：**把调度这件事从 CPU 搬到 GPU 上**——用原子计数器做动态工作队列、把整条流水线录成 CUDA Graph 一键重放、让 kernel 自己启动下一个 kernel，多卡场景用 NVSHMEM 让 GPU 之间直接读写彼此的显存。

---

## 1. 敌人：不均匀的工作 → SM 空转

- 输入相关的循环让不同线程干的活不一样多：快的 block 早早收工，它的 SM 就闲着等最慢的那个。
- 大白话：**给每个工人发一摞固定的任务，手气好的早下班、手气差的加班——但整个厂房的电费你照付**。
- 怎么诊断：Nsight Systems 看时间线空档；Nsight Compute 的 *SM Active %*（有活跃 warp 的周期占比）；配合 NVTX 标注定位到代码。

## 2. 原子计数器与动态工作队列

### 原子操作快，但怕"挤"

- 现代 GPU 的全局原子操作在 **L2 缓存里完成**（片上，不用绕 DRAM），无竞争时非常快。
- 竞争时会**串行化**：所有线程在同一个地址排队。
- 观测指标：Nsight Compute 的 `atomic_transactions_per_request`——**≈1.0 理想**；>1.0 说明线程在反复重试。

### 解法：按批领任务（amortize）

```cuda
// 改造前：每个任务一次 atomicAdd
int idx = atomicAdd(&queue_head, 1);
// 改造后：一次领 32 个（一个 warp 的量）
int start = atomicAdd(&queue_head, 32);
```

- 实际写法：**warp leader**（用 `__ffs(__activemask())` 选出）一人做 atomicAdd，再用 `__shfl_sync` 广播起始下标给全 warp，32 人并行处理。原子操作从"每线程一次"降到"每 warp 一次"。
- 大白话：**别让每个工人为每个任务都去售票窗口排队——每个班组派一个跑腿的，一次领 32 张票**。
- 批量 8 或 16 通常就能消掉绝大部分竞争。

### 动态队列的效果

- 完成任务的 warp **立刻去领下一批**，直到队列取空（`idx >= N` 退出）——没有 SM 提前"下班"。
- 书中微基准：静态分配 ~200 ms → 动态队列 ~100 ms（**2×**，极端不均匀场景）；一般不均匀场景 **10–20%**。轻微不均匀时原子+shuffle 的开销可能得不偿失——**要实测**。
- PyTorch 没有对应的高层 API，这是纯自定义 CUDA kernel 的技术。

## 3. CUDA Graphs：录一次，重放一万次

### 为什么

- 每次迭代逐个 launch kernel/memcpy，都要付 **CPU 启动开销**（每个几微秒）+ 主机-设备握手。
- CUDA Graph 把整条流水线（节点=kernel/拷贝，边=依赖）**录制一次**，之后**一次主机调用整体重放**。
- 大白话：**别每天晚上口述菜谱，把烹饪节目录下来，以后按播放键**。
- 谁在用：PyTorch（`torch.cuda.CUDAGraph`、`torch.compile(mode='reduce-overhead')`）、vLLM、TensorRT-LLM（按 batch size 分桶，每桶预录一张图，运行时挑图重放）。

### 怎么录（C++）

```cuda
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernelA<<<..., stream>>>(...); kernelB<<<...>>>(...); kernelC<<<...>>>(...);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
for (...) cudaGraphLaunch(instance, stream);   // 每轮只有 1 次主机调用
```

### PyTorch 版的关键规矩

- **必须先 warm-up 跑一遍**：capture 不会录制 cuBLAS/cuDNN 的懒初始化，不热身可能录制失败。
- **所有张量地址、形状必须固定**：提前分配 static 张量，图内用 `out=` 写入；每轮新输入用 `static_x.copy_(new_X)` 灌进去，**绝不能重新分配**。
- `torch.cuda.graph_pool_handle()`：专用内存池保证指针稳定。
- capture 内不能有：内存分配、`print`、RNG 初始化、主机回调。
- 大白话：**图是 GPU 工作的录像带，演员（张量）每一条都必须站在同一个标记点上**。

### 效果（书中示意，Table 12-1）

| 每 100 轮 | 之前 | 之后 |
|---|---|---|
| CPU launch 次数 | 300 | 100 |
| 主机同步次数 | 300 | 0 |
| kernel 间 GPU 空转 | ~3 µs/轮 | 0 |
| 单轮延迟 | ~1.00 ms | **~0.75 ms（快 25%）** |

- 注意：图**不会让单个 kernel 变快**，它消灭的是 kernel **之间**的开销。kernel 越小越多（正是 LLM decode 的形态），收益越大。

### 动态更新：别重录，打补丁

- 形状/指针变了 → `cudaGraphExecUpdate`（或 `cudaGraphExecKernelNodeSetParams`）直接改已实例化的图：**几微秒**搞定，保住亚 100µs 的重放路径。
- 典型推理流程：按**最大 batch（如 128）录模板图** → 来了 batch 64 就 update 启动维度和指针 → 重放。
- 不能增删节点——结构变了就得重录（recapture），之后继续用 update。
- 大白话：**长期包餐合同改一下人数就行，不用每天重新谈合同**。

## 4. 设备侧发起的图启动（Device-Initiated Graph Launch）

- 让**正在运行的 kernel 直接启动一张预录的图**，CPU 完全不参与。
- 前提：`cudaGraphInstantiate(..., cudaGraphInstantiateFlagDeviceLaunch)` + **先 `cudaGraphUpload` 上传到 GPU**（否则报错）。
- 收益：启动延迟约为主机侧图启动的 **1/2**，且**不随图的节点数/宽度增长**（主机侧会随之涨）。
- 三种模式：

| 模式 | 语义 | 限制 |
|---|---|---|
| Fire-and-forget | 子图**立即**执行，与父 kernel 并行，父不等它 | 一次图执行最多 120 个 |
| Tail launch | 父 kernel **结束后**才执行（continuation 续接） | 最多 255 个排队；自我 tail-launch 同时只能挂 1 个 |
| Sibling | fire-and-forget 变体，作为父图的**平级**，不挡父图的 tail | — |

- 选择：子图结果要被后续用 → **tail**；独立旁路任务 → fire-and-forget/sibling。
- **杀手级模式（LLM decode 常驻调度器）**：把 transformer block 前向录成图；一个轻量常驻 kernel 用 `atomicAdd(&queueHead,1)` 领任务 → **tail-launch** decode 图 → 图跑完回来领下一个。**整条 token 流水线全程不碰 CPU**。这就是 GPU-resident 推理引擎的骨架。

## 5. 条件图节点（Conditional Graph Nodes）

- 传统图在录制时就定死了，分支逻辑得弹回主机做。条件节点把 **IF / IF-ELSE / WHILE / SWITCH 直接嵌进图里**，GPU 自己评估"条件句柄"并选择子图。
- 设置条件：设备 kernel 里 `cudaGraphSetConditional(handle, flag)`——**只用一个线程写**，避免竞争。
- 可以**嵌套**（WHILE 里套 IF），多级决策零 CPU 往返。
- 大白话：**录好的烹饪节目里加上"剧情分支按钮"，而且是厨房（GPU）自己按，不用导演（CPU）插手**。
- 现状：PyTorch 还没有 Python API，要用 C++ 自己集成。

## 6. 动态并行（Dynamic Parallelism, DP）

- 图适合**事先已知**的流程；当**工作形态由数据决定**（图遍历、自适应网格、层级归约），录不出图——这时让父 kernel 检查自己的输出、**在设备上直接 launch 子 kernel**（编译要加 `-rdc=true`）。
- LLM 例子：大多数 token 走标准路径，个别 token 需要辅助 attention——DP kernel 运行时只为这些位置追加启动。
- 识别信号：profiler 时间线上 `Kernel A → GPU 空档 → Kernel B`，空档就是 CPU 在做决定。
- 语义要点：**父 kernel 隐式等待所有子 kernel 完成**，设备侧不需要（也不该用）`cudaDeviceSynchronize()`；主机只需同步一次 stream。
- 成本与限制：
  - 子启动消耗**栈空间**（`cudaDeviceSetLimit(cudaLimitStackSize, ...)`）；
  - 默认**最多 2,048 个待处理子启动**（`cudaLimitDevRuntimePendingLaunchCount` 可调）；
  - 设备侧启动开销 ≈ 主机侧（~20–25 µs 量级）——**子任务太小会亏本**。
- 效果（书中示意，Table 12-2）：GPU 空转 40%→5%，总时间 1.00→0.75 ms。附赠好处：中间结果**不离开显存**，局部性更好。
- **选型口诀：DP 管"料不到"的活，图管"录得下"的活**。固定重复流程优先 device-launched graphs（能摊销），别为它付 DP 的每次启动税。

## 7. 多 GPU / 跨节点编排

### 总原则：通信必须被计算盖住

- NVLink 再快也不如本地 HBM。**没有重叠，扩展就在"通信时间=计算时间"处见顶**；有重叠，4 卡才能像一张 4 倍速的卡。
- 背景：GB200 NVL72 一个机柜 = 72 张 Blackwell + 36 个 Grace，一个 NVLink 域、统一寻址、最多 30 TB 内存——硬件已经把集群做成"一张大 GPU"，剩下靠软件。

### 工具箱

| 工具 | 用途 | 关键点 |
|---|---|---|
| `cudaMemcpyPeerAsync` | 卡间大块搬运 | 放在单独的通信 stream 上后台跑 |
| NCCL | 集合通信（all-reduce 等） | 非阻塞、自动组 ring/tree；PyTorch DDP 的底座 |
| CUDA-aware MPI | 跨节点点对点 | 直接传显存指针，走 GPUDirect RDMA，不经过主机内存 |
| NVSHMEM | 细粒度、事件驱动的卡间共享 | 见下 |

### NVSHMEM：GPU 直接读写对方显存

- **PGAS 模型**：每张 GPU 是一个 **PE**，共享一个"分区全局地址空间"；**设备代码**就能对远端显存做单边 put/get。
- 经典 send-and-signal 三步：`nvshmem_float_p`（写数据到对方）→ `nvshmem_quiet()`（确保写完）→ `nvshmem_int_p`（升旗）；接收方 `nvshmem_int_wait_until` 等旗升起再读。
- 大白话：**不再通过邮局（CPU）寄包裹，GPU 直接走过去把东西放到对方桌上**。多步通信变成一次硬件事务，接近线速。
- 三个实用模式：
  1. **两段流水线**：GPU 0 算 attention → put + signal → GPU 1 常驻 kernel 接着算 MLP，GPU 0 已开始下一个 batch——交接被计算掩盖；
  2. **设备侧 work stealing**：`nvshmem_int_atomic_inc(queue_head)` 全集群抢任务；
  3. **锁步执行**：`nvshmemx_collective_launch()` 同时在所有 PE 上启动协作 kernel（用设备侧 barrier/collective 的 kernel **必须**这样启动），内部 `nvshmem_barrier_all()`。
- 警告：**别过度同步**——全局 barrier 让所有卡等最慢的那个；能用点对点 signal 就不用 barrier_all。

### NCCL + CUDA Graphs

- NCCL 集合通信可以像普通 kernel 一样**录进图**：forward → ncclAllReduce → backward 录成一张图，每步一次 `cudaGraphLaunch`——CPU 负载和 launch 抖动（jitter）都降。
- **铁律：所有 rank 必须用同一个 communicator、以完全相同的顺序录制和重放**。错配轻则死锁（好查），重则静默算错（难查）。录制前先跑 warm-up 集合通信把 communicator 初始化好。
- **Bucketed all-reduce**：把梯度分桶，通信 stream 和计算 stream 在图里交错——通信几乎全部藏进计算。PyTorch DDP 自动做了变体，图化可以再挤掉 CPU 开销。规模越大（几万卡），这些节省越是复利。

## 8. Roofline 指导选型（本章工具怎么挑）

| 你的 kernel 在 roofline 上的位置 | 该用什么 |
|---|---|
| **Memory-bound**（脊点左侧） | 重叠为王：异步拷贝、多 stream、并发 kernel；**降精度**（FP16/8/4）减少字节数。算子融合帮助有限 |
| **Compute-bound 但没到顶** | 消灭启动开销：算子融合、persistent kernel、CUDA Graphs、设备侧启动 |
| **两不沾（中间地带）** | 提高并发度：多 kernel/stream/图并行跑，把聚合工作点往两轴上推 |

- 流程：用 Nsight Compute 数 FLOPs 和字节 → 画点 → 选工具 → **改一处测一次**。Roofline 指方向，profiling 定结论。

## 9. 容易被问到的点（答辩预备）

1. **CUDA Graph 会让 kernel 本身变快吗？** 不会。它消除的是 kernel 之间的 CPU 调度/启动开销和空档，所以小 kernel 密集的流水线（LLM decode）收益最大。
2. **图里能改 batch size 吗？** 能，`cudaGraphExecUpdate` 改参数/指针（微秒级）；改图结构（增删节点）必须重录。
3. **DP 和 device-launched graph 怎么选？** 工作形态运行时才知道 → DP；固定重复流程 → 图（开销可摊销）。
4. **为什么 PyTorch 图要求 static 张量？** 重放时按录制时的指针执行，地址变了就是未定义行为——所以用专用内存池+`copy_` 更新输入。
5. **NVSHMEM 和 NCCL 什么关系？** NCCL 管"整齐"的集合通信（all-reduce/broadcast）；NVSHMEM 管"不整齐"的细粒度单边通信和设备侧同步（动态负载均衡、事件驱动）。互补，不互替。
6. **tail launch 怎么实现 GPU 上的无限循环？** kernel 用 `cudaGetCurrentGraphExec()` 拿到自己的图句柄，再 tail-launch 自己——GPU 常驻调度器由此而来。
