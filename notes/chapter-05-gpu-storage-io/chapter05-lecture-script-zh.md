# 第五章讲稿（对应 chapter05-presentation.pdf，共 31 页 / 约 50 分钟）

> **结构**：五部分、单封面，紧跟原书小节顺序（Fast Storage → GDS 家族 → 分布式文件系统 → 数据管线 → 持续调优工作流）。本章只有 28 页原文，比第六章短，目标 30–35 页内讲完。
> **时间分配**：开场 1–2 页 3 分｜Part I 3–7 页 8 分｜Part II 8–14 页 12.5 分｜Part III 15–19 页 8 分｜Part IV 20–25 页 10 分｜Part V 26–28 页 6 分｜收尾 29–31 页 2.5 分 = **50 分**。
> **超时保险（按序启用）**：① 第 13 页 cuda-checkpoint 压到 30 秒（标了星号的旁支页）；② 第 6 页 io_uring 细节一句话带过；③ 第 27 页右栏四条并成两条讲。
> 每节结构：🎤 口播稿／大白话收束句／❓ 预备问答；🗣 = 必说的转场或补充。

---

## 第 1 页｜标题页

🎤 大家好，今天讲第六章之前的第五章：GPU-Based Storage I/O Optimizations——面向 GPU 的存储与 I/O 优化。上次第六章我们钻进了 GPU 内部；这一章往回退一步，回答一个更朴素的问题：**数据怎么才能足够快地喂进 GPU**。这章比第六章短，我们大概 50 分钟讲完。

---

## 第 2 页｜Roadmap: Five Parts, Each Answers One Question（路线图：五部分各答一问）

🎤 还是老规矩，路线图是**问题清单**：**Part I 存储基础**——数据怎么摆、怎么读，磁盘才能跑出峰值带宽；**Part II 直达 GPU 的数据通路**——数据怎么绕过 CPU 的中转缓冲直接进显存（GDS 一家子）；**Part III 大规模共享存储**——什么东西能同时喂饱很多节点而自己不成为瓶颈；**Part IV 数据管线**——DataLoader 和预处理怎么保证每块 GPU 都吃饱；**Part V 持续调优工作流**——任务到底卡在 I/O、通信还是计算，以及怎么让好状态保持下去。
全章一句话（金句块）：**Keep the data pipeline as full as the compute pipeline——让数据管线和计算管线一样满；饿着的 GPU 就是白买的 GPU。**
🗣 后面出现 io_uring、cuFile、3FS、DALI 这些名词时，随时问自己"它在回答哪个问题"——一定在这五个里。

---

## 第 3 页｜Part I — Feeding Data Is as Important as the Compute Itself（喂数据和算力同等重要）

🎤 开场先立场景：一个 **100 万亿参数**的模型在几千块 GPU 上训练，要吃掉**几十亿个样本**——语言模型是 TB 级文本，视觉模型是 PB 级图像。如果存储管线慢了，**GPU 就会挨饿、空转**——第四章做的所有通信优化都救不了它。
右框算一笔账（书中例子）：假设每块 GPU 需要 **200 MB/s** 的训练数据才能吃饱。8 块 GPU 就是约 **1.6 GB/s**；一台 GB200 NVL72 机柜里 72 块 Blackwell，就要 **14–20 GB/s** 的聚合存储吞吐。注意小字：如果你的样本是重型多模态数据，要用**实测的"每样本字节数 × 每秒样本数"**来标定——需求可能高得多。
本章三件事：高效地从本地或远端存储读数据、预处理它、并把 I/O 和 GPU 计算**重叠**起来。

大白话：**全世界最好的后厨，备菜线断了照样上不了菜——喂料本身就是一份全职工作。**

---

## 第 4 页｜Data Locality: Put the Data Physically Close to the Compute（数据局部性）

🎤 第一原则：**把数据放得离计算物理上越近越好**。"近"的意思是：最好在**本机的 NVMe SSD** 上；退一步至少**同机柜**，用 NVMe-over-Fabrics（NVMe-oF）这种机柜内高速互连——网络跳数越少，性能越稳定。
如果数据在网络存储上（NFS、Amazon S3），要验证**所有计算节点的聚合读带宽**能不能覆盖上一页算出来的胃口。并行文件系统（Lustre、GPFS）可以把数据**缓存到本地 SSD**。
右框是常用做法：**按节点切分数据集**。100 TB 数据、10 个节点，就预先给每个节点切 10 TB 到本地盘；每个节点的 loader **只读自己那份**——不再有冗余的网络读。PyTorch 的 **DistributedSampler** 会协调各进程，每个 epoch 拿到**互不重叠的切片**，正好和按节点分片对齐。

大白话：**别让每个车间都向海外仓下单——把它要用的零件提前上到它自己的货架。**

❓ 可能被问："数据放不下本地盘怎么办？"→ 这正是 Part III 的话题：并行文件系统 + 本地缓存，或对象存储 + 预先 staging。

---

## 第 5 页｜Sequential Beats Random: Shard Your Samples(顺序读完胜随机读)

🎤 第二原则：**顺序读远快于随机读**——存储设备对大块连续读的吞吐远高于小块随机读，GPU 也喜欢大块连续的数据。
反面教材：**几百万个独立的小图片文件**——磁盘上到处随机寻址。正确做法：把样本打包进**少量大的二进制分片**——Arrow、TFRecord、Parquet、WebDataset tar——一次读取自然带回**很多样本**。对象存储同理：把小的 S3 对象**提前合并**成大对象。
右框：**读取粒度也要调**。1 MB 一读比 4 KB 一读吞吐高——每次读的固定开销被摊薄了。很多 loader 库暴露 buffer 大小和 prefetch 块大小，虽然它们内部已经做了优化，**这些参数仍然值得调**。
最后一条是书里的警告：GPU 越快，小随机读**越早**成为瓶颈——现代并行文件系统虽然能扛一定的小随机读，但要**显式验证**它的表现。

大白话：**去仓库一趟就拉一整托盘，别一次拿一颗螺丝——而且螺丝要在开训前就装托盘。**

---

## 第 6 页｜When Reads Must Be Random（不得不随机读的时候）

🎤 有的访问模式就是随机的，怎么办？答案：**并行**。同时发出多个读请求：多线程调 `pread()`，或者用 Linux 的异步 I/O 接口 **io_uring**——它支持预注册缓冲区和 polling，能以极低的内核开销**成批提交 I/O 请求**，用高并发把延迟藏起来、把 IOPS 顶上去。
注意左下：带缓冲的 I/O 和大 read-ahead 只帮**顺序**扫描；随机读收益很小——要么**把随机读攒成大块**，要么用现成的数据集 API（TFRecordDataset、PyTorch IterableDataset + DataLoader）。
右框是文件系统层面的建议：Linux NVMe 服务器上常用 **XFS**，挂载时加 **noatime**——不然每次读还要写一次访问时间。云上的 Amazon EFS：用 **Max I/O** 性能模式；要稳定带宽就从 Bursting 切到 **Provisioned throughput**。

大白话：**真要一颗颗取螺丝，就一次派一群跑腿的，并且别再给每趟打卡盖章（noatime）。**

---

## 第 7 页｜NVMe and Kernel Tuning: Scheduler, Read-Ahead, RAM（NVMe 与内核调优）

🎤 再往下一层，内核。三个旋钮：
① **I/O 调度器**：现代 Linux 用多队列的 blk-mq，把 I/O 摊到各 CPU 核。NVMe 用 **none** 或 **mq-deadline**（老的 CFQ 已废弃）；`/sys/block/nvme*/queue/scheduler` 一查便知——一般出厂就是对的，但值得花十秒确认。
② **Read-ahead**：内核检测到顺序读会自动预读，默认约 128 KB；流式读大文件的话，用 `blockdev --setra` 提到**几 MB**——syscall 更少、读取流水线化。
③ **接口和条带**：确认 SSD 插在最快的接口上、PCIe 通道够用；单盘喂不饱 GPU 时，多盘 **RAID 0** 条带化。
右框是**内存捷径**：Linux page cache 自动缓存最近读过的数据——中等规模数据集热缓存收益很大（PB 级会把缓存冲爆）。如果数据集**装得进 RAM**（包括 Grace Blackwell 的 CPU+GPU 统一内存），**启动时整个预加载进内存**——超高速内存缓存，训练期间几乎零磁盘 I/O。

大白话：**内核本来就会帮你预读——你只需要告诉它读多远；要是整个货品目录塞得进前厅，就搬进前厅。**

---

## 第 8 页｜Part II — GDS: Cut the CPU Bounce Buffer Out of the Path（GDS：砍掉 CPU 中转）

🎤 磁盘调好了，下一个问题：为什么每个字节还要**绕道 CPU 内存**？普通读取路径：SSD → **CPU 内存** → 一次 CUDA 拷贝 → GPU 显存——中间多了一跳"host bounce buffer（主机中转缓冲）"。
**GDS（GPUDirect Storage）**：GPU 直接对 SSD 或网卡发起 **DMA**，数据**直接落进 HBM**。支持本地 NVMe 和 NVMe-oF。
两个必须说清的点：① **CPU 仍然负责配置和编排每一次 I/O**——消失的是经过主机内存的那次拷贝，不是 CPU 的控制角色；② 右框，GDS 和 GPUDirect RDMA 是**兄弟**：GDS 加速**存储→GPU** 的 DMA，GPUDirect RDMA 加速**网络→GPU** 的 DMA；都去掉 bounce buffer，都**不**去掉 CPU 编排。

大白话：**货车现在直接卸到工位，不再经过前台——但单子还是前台签的。**

---

## 第 9 页｜GDS in the Wild: VAST's Before/After Architecture（实测架构图）

🎤 看图（Figure 5-1）。右半边"Without GDS"：传统的**分段 DMA**——粉色路径经过主机内存来回倒。左半边"With GDS"：GPU 经过 PCIe switch **直接拉数据**——主机内存拷贝消失，CPU 占用下降。
数字：VAST 报告在 A100 上顺序读吞吐 **+20%**；在 H100 上 **+30% 以上**——NIC 带宽越高，原来压在 CPU 上的搬运负担越重，去掉后收益越大。
书的告诫（最后一条）：**在你自己的负载上验证**——收益随 I/O 大小、队列深度、网卡代际、文件系统实现而变。

大白话：**抄近道省多少时间，取决于原来前台有多堵——量你自己的楼，别信开发商的样板间。**

---

## 第 10 页｜Using GDS in Practice: the cuFile API（实际怎么用：cuFile）

🎤 应用侧通过 CUDA 的 **cuFile** 库走 GDS：**cuFileRead** 直接作用在普通的 **POSIX 文件描述符**上，自动处理缓冲区对齐，内核里由 **nvidia-fs** 驱动编排 DMA。异步版本 **cuFileReadAsync / cuFileWriteAsync** 把存储 I/O 挂到 **CUDA stream** 上（stream 第 11 章细讲）——从而能和计算重叠、流水线化。
一个容易搞混的点：存储→GPU 这条路**不使用主机 pinned memory**——cuFile 注册的是 **GPU 设备缓冲区**。
右框是前提条件：新的 NVIDIA GPU + 驱动 + CUDA toolkit，加上支持 DMA 的存储栈：本地 NVMe / NVMe-oF 配 **XFS/EXT4 + O_DIRECT**、**NFS over RDMA**、以及集成了 nvidia-fs 的并行文件系统（BeeGFS、WekaFS、VAST、IBM Storage Scale）。
书的提示：尽量用 **O_DIRECT**；新版 cuFile 也能处理非 O_DIRECT 的文件描述符，但**不对齐可能带来额外拷贝**。

大白话：**一个库调用，把叉车路线从头订到尾——前提是卸货码头（文件系统）说同一种协议。**

---

## 第 11 页｜When Does GDS Actually Help?（GDS 什么时候真有用）

🎤 这页讲书里少见的"诚实话"：如果你的 CPU 本来就**轻松搞定**数据搬运，GDS 对吞吐可能改变不大——但它仍然**降低 CPU 占用**，腾出核去做预处理。反过来，如果 CPU 被 memcpy **打满了**，GDS 帮助巨大。
左下算例：每秒 1,000 个 1 MB 的 batch = **1 GB/s**。用 CPU 搬要吃掉**好几个核**；GDS 让 GPU 自己拉。速率更高、GPU 上千块时，效果成倍放大。
右栏三点：① 训练负载**压倒性地以读为主**，所以 GDS 的收益基本都在读上测；② **checkpoint 写**也要快——RDMA 加速的**写**需要文件系统支持，比如 **WekaFS** 的 GDS 插件读写都覆盖；③ 存储厂商生态（WekaIO、DDN、VAST、Cloudian……）让企业级 NAS/并行文件系统**开箱即用** GDS。

大白话：**近道最值钱的时候，是前台职员本身就是排队原因的时候——即使不更快，职员也被解放出来干别的了。**

---

## 第 12 页｜Measuring It: gdsio Before and After（gdsio 实测）

🎤 怎么验证？NVIDIA 自带的 **gdsio**，默认装在 `/usr/local/cuda/gds/tools`。左边两条命令，**唯一区别是 `-x` 传输选择器**：`-x 2` 走 CPU 路径（存储→CPU 内存，pinned host memory + 异步拷贝）做基线；`-x 0` 走 GDS 路径——文件、worker 数、总量、I/O 大小全部一样，**对比要公平**。
结果（Table 5-1）：吞吐 **8.0 → 9.6 GB/s（+20%）**，平均延迟 **1.25 → 1.00 ms（−20%）**——同时省下了原来在主机缓冲里搬数据的 CPU 周期。
🗣 这就是这一章的方法论缩影：**先跑个基准，再谈架构。**

大白话：**一个工具、两个 flag、一个干净的 20% 结论——吵架构之前先跑它一遍。**

---

## 第 13 页｜Checkpointing GPU State with cuda-checkpoint *（旁支页，30–60 秒）

🎤 **这页标了星号，是存储故事里的一个旁支**——不是写模型 checkpoint，而是给**整个 CUDA 进程**拍快照。**cuda-checkpoint** 配合 CPU 侧的进程快照工具 **CRIU**：挂起时锁住 CUDA 入口、**排干在途工作**、把显存拷到主机内存、释放 GPU——然后 CRIU 就能把整个进程落盘。Driver API 四件套：`cuCheckpointProcess{Lock, Checkpoint, Restore, Unlock}`；恢复需要 persistence mode，而且可以**换到同型号的其他 GPU** 上恢复。挂起耗时 ≈ **显存镜像大小 ÷ 主机链路带宽**。
右框两个澄清：① 这条路**不是 GDS 路径**——显存先进主机内存，再由 CRIU 落盘；② 它和框架级 checkpoint（PyTorch state-dict）**互不替代**——用途是容错、抢占、长任务迁移，是**补充**。

大白话：**它是把整个车间连人带机器冻结、搬去别处重开——和复印图纸（模型 checkpoint）是两码事。**

---

## 第 14 页｜DeepSeek's Fire-Flyer File System (3FS)（DeepSeek 的 3FS）

🎤 讲个业界案例。DeepSeek 从零造了开源文件系统 **3FS**，出发点是一个观察：AI 负载做**海量随机读**——传统的读缓存不但没用，甚至**帮倒忙**。所以 3FS 干脆**不做缓存、直接 I/O**，每个请求直达 NVMe，绕过内核 page cache。
右图（Figure 5-2）：四个组件——cluster manager、metadata service、storage service、client——全部跑在 **RDMA 网络**（InfiniBand/RoCE）上。
数字：68 节点集群上聚合读 **6.6 TB/s**（同时还扛着 1.4 TB/s 的后台流量），环境里最高报过 **7.3 TB/s**——相似硬件上 Ceph 只有约 **1.1 TB/s**。
右下 FUSE 警告（书里的坑）：**用户态（FUSE）文件系统给不了 GDS 路径**——GDS 需要内核级的 O_DIRECT 集成；要真 GDS，用内核客户端（NVMe/NVMe-oF、BeeGFS、WekaFS、IBM Storage Scale、VAST）。
最后一条别漏：自建文件系统是**巨额投入**——大多数团队应该从现成系统起步，这就是下一部分。

大白话：**DeepSeek 按 AI 的购物方式重造了仓库：随机抓取、不设展示架、只留叉车。**

---

## 第 15 页｜Part III — NFS: Simple to Set Up, Simple to Saturate（NFS：好搭也好堵）

🎤 自建文件系统是例外，我们大多数人挂什么？先说最简单的 **NFS**：一台共享服务器、所有节点读同一个地方——方便，但很多节点同时读时就是**吞吐瓶颈**。
必须用 NFS 的话：服务器配**多块快网卡**，或者干脆**多台 NFS 服务器**各管一部分数据；底下用 NVMe（甚至 RAID 0）。书的结论：NFS **只适合小规模**（几个节点）——更大的集群要并行文件系统或云缓存（下一页）。
右边是真正有用的客户端调优：**rsize/wsize 拉到最大**（1 MB）、`noatime` 干掉访问时间写、`actimeo=60,lookupcache=pos` 缓存文件属性和目录项 60 秒——几行挂载参数，大幅摊薄每请求开销。

大白话：**一个柜台服务全城没问题——直到全城同时上门。**

---

## 第 16 页｜Parallel Filesystems and Object Stores（并行文件系统与对象存储）

🎤 左边**并行文件系统**（Lustre、GPFS）：为高并发高吞吐而生。Lustre 用多个 **OST**（对象存储目标）出数据；把大文件**条带化**跨 OST 存（`lfs setstripe`，比如 4–8 个）：4 个 OST × 每个 500 MB/s = 单文件理论峰值 **2 GB/s**。训练期间要监控（lmt、厂商面板）：少数存储节点**发热**通常意味着**分片不均**——要查原因。
右边**对象存储**（S3）：训练时裸读很慢。两条路：训练前**staging 到本地 NVMe**（用高并行工具：s5cmd、AWS C++ SDK），或者在 S3 上面加**缓存层**（Amazon FSx for Lustre）。直接流式读的话，用**大的 range GET**、多线程。
书的提醒：缓存层要**验证性能提升对得起成本**——数字不达预期就直接找云厂商对架构。

大白话：**十个装卸码头胜过一个；货在海外（S3）的话，要么提前发货，要么租个本地中转仓。**

---

## 第 17 页｜Two Blunt but Effective Levers: Replicate and Compress（两个笨但有效的杠杆）

🎤 两个"暴力但常用"的办法。左：**全量复制**——把整个数据集拷到每个节点的本地盘。**零网络读**、立竿见影，代价是存储空间。右：**压缩存储、读时解压**——JPEG、压缩的 Arrow/Parquet：用 CPU/GPU 周期换 I/O 带宽。什么时候划算？**恰好 I/O 是瓶颈、计算闲着**的时候。
下面三条是 GPU 侧的新武器：**nvJPEG** 在 GPU 上解码图片；Blackwell 片上的 **Decompression Engine** 硬件解压 **LZ4、Snappy、Deflate**，在流水线内完成——**SM 腾出来跑计算 kernel**，I/O 受限的负载优先选这几种格式。**NVLink-C2C**（双向约 900 GB/s）保证 CPU 协助的环节不成为新瓶颈。
底线：如果**解压时间取代 I/O 成了新瓶颈**，这个压缩就不值。

大白话：**要么给每个车间发一整套目录，要么把目录真空压缩了寄——但拆包不能比寄件还慢。**

---

## 第 18 页｜Monitoring Storage I/O: the Toolbox（存储监控工具箱）

🎤 老规矩：不测量、不下结论。左：主机侧 **iostat、iotop、nvme-cli、perf、eBPF** 加厂商面板——看队列、延迟、read-ahead 效果、缓存命中率；本地 NVMe 或通往 NAS/对象存储的网络链路**打满了没有**。**DCGM** 补上 GPU 侧的 I/O 统计——合起来才能看清"GPU 因 I/O 挨饿"这件事。
右：GDS 感知的追踪——Nsight Systems 加 **`--trace=gds`**，把 **cuFile API 活动**画上时间线；`/etc/cufile.json` 里可以打开 cuFile 静态 tracepoint。注意坑：NVMe **点对点 DMA** 的内核态计数器 Nsight 里看不到，部分 GDS 栈上也没有。

大白话：**跟每一章一样：没有仪表就没有裁决——磁盘、网卡、GPU 各配各的表。**

---

## 第 19 页｜Whose Fault Is the Idle GPU? A Two-Timer Decomposition（两只秒表拆责任）

🎤 GPU 在等数据——但等的是谁？PyTorch 里给 **`next(data_iterator)`** 计时，量到的是 **GPU 等下一个 batch 的总空闲时间**——包括后台 prefetch 和 H2D 拷贝，不只是 Python 逻辑。
把它拆成两半：**(1) Python 侧成本**——把 `num_workers=0`（没有预取了），单独计时裸的迭代器拉取；**(2) H2D 拷贝成本**——用 **CUDA events** 包住 `.to("cuda")`，或看 Nsight Systems 的 **Copy 泳道**。两个数对比总空闲时间，就知道该加速 **Python 管线**（加 worker、简化 transform）还是**传输路径**（pinned memory、更快互连、GDS）。
右框是为什么值得做（书中例子）：GPU **30%** 的时间在等数据 = 吞吐被 I/O 封顶；调完只等 **5%**——训练 steps/s **成比例上涨**。存储调优就是消灭一堆小低效——这里 5 ms、那里一个太小的 buffer——规模一大，全都**加起来算账**。

大白话：**先量出全组人站着发呆多久；再把责任拆给后厨（Python）和送货车（H2D 拷贝）。**

---

## 第 20 页｜Part IV — The Data Pipeline: The Last Mile to the GPU（数据管线：最后一公里）

🎤 存储只负责把字节送到；还得有人把字节变成 batch。典型加载路径四步：**读**（存储）→ **解码/反序列化**（JSON、JPEG）→ **变换**（分词、裁剪）→ **collate 成 batch**。这些步骤吃 CPU——但计算重的可以**卸载到 GPU**。目标只有一个：GPU **永不空等**，同时 CPU 上恰好有**合适数量**的活并行跑着。
右框给个视角（书原话的量级）：调差的输入管线能浪费 **50%** 的 GPU 时间；而算法层面的小聪明常常只挣**几个百分点**——优化管线经常**比改模型更划算**。

大白话：**最后一公里决定交货日期——仓库堆满零件，装配台慢照样出不了货。**

---

## 第 21 页｜DataLoader Essentials: the Five Knobs（DataLoader 五个旋钮）

🎤 表格五行就是五个旋钮：**num_workers**——并行取数+预处理的**进程**（绕开 Python GIL），数量靠实测；**pin_memory=True**——页锁定的主机缓冲，是真异步 H2D 的前提；**non_blocking=True**——异步 `.to(device)`，和 pinned 配对使用；**persistent_workers=True**——worker 跨 epoch 存活，省掉反复拉起进程的开销；**prefetch_factor**——每 worker 预取几个 batch（默认 2），I/O 突发的负载可以提到 4–8。
下面四条：worker 太少 → GPU 空转；太多 → 抢 CPU 核和 I/O 带宽。目标状态：**磁盘吞吐接近 100%、CPU 留有余量**。高核数 CPU（72 核 Grace）可以多开 worker——注意收益递减。**扩 GPU 必须同步扩管线**：每个 rank 读自己的分片，聚合加载量随集群规模增长——否则瓶颈直接转移到数据摄入。最后：`ulimit -l`（memlock）调高，否则大 pinned 缓冲会分配失败。

大白话：**一台机器五个旋钮——而且多雇大厨（GPU）就必须多雇备菜工（worker）。**

---

## 第 22 页｜The Overlap in Code: Pinned Memory + Prefetch + Streams（重叠的代码长相）

🎤 这页把上一页的旋钮拼成能跑的代码。DataLoader：8 个 worker、每个预取 **4 个 batch** 到 **pinned** 内存（总共 num_workers × prefetch_factor 个在队列里）。下半段：单独一条 **copy stream** 做 `to(device, non_blocking=True)`；compute stream 先 `wait_stream(copy_stream)` 确保 H2D 完成，再跑模型。
效果：GPU 在算第 N 个 batch 时，第 **N+1** 个**已经在传输路上**。关键因果链别讲反：**因为源内存是 pinned 的，DMA 才能真正异步**；不 pin 的话，系统每次传输都要**现场临时 pin 一遍**——这就是没点单的延迟。

大白话：**备菜台永远提前摆好下一托盘——大厨两道菜之间不用抬头。**

❓ 可能被问："wait_stream 不是又同步了吗？"→ 它是 **stream 间的顺序约束**，不是全局同步：只让 compute stream 上后续 kernel 等这次拷贝，CPU 和其他 stream 照常跑。第 11 章细讲。

---

## 第 23 页｜Avoid Python Bottlenecks; Profile the Loader in Isolation（Python 陷阱与隔离测量）

🎤 左边两个红旗：① **纯 Python 循环逐行分词**——改成向量化操作，或用 **Rust/C++ 打底**的库（Hugging Face Tokenizers、TorchText）——Python 只是接口皮；② 变换尽量**按 batch 做**：先用自定义 `collate_fn` 拼起来、再对整个张量做向量化变换——但有些操作天生只能逐样本，先搞清楚负载再决定。隐藏瓶颈很容易混进来：**调试日志**、昂贵的 CPU 变换——常常**只在压力下**才显形。
右边是隔离测量法：关掉所有 GPU 工作，单独计时 DataLoader 产出 **100 个 batch** 的耗时，和目标迭代时间、实测 GPU 空闲时间对比。loader 太慢 → 删日志、简化变换、**加 worker**。
书的小注意：关掉 GPU 工作的同时也去掉了 **kernel 启动开销**，所以隔离出来的 loader 吞吐会显得**偏低**——方法依然有用，心里有数就行。

大白话：**关了堂食单测备菜速度——但记住正式营业时前厅自己也有噪音。**

---

## 第 24 页｜Multimodal Preprocessing on the GPU: NVIDIA DALI

🎤 预处理太重怎么办？**DALI** 把解码和增强搬上 GPU（或用优化过的 C++ CPU 代码）：JPEG 解码、随机裁剪、resize、归一化全在 GPU 上做，用的是 GPU 的**媒体加速硬件**。编程模型是**声明式静态图**：继承 `Pipeline`、在 `define_graph()` 里声明算子；执行、预取、线程池 DALI 自己管。
书中例子：目标检测管线，CPU 跑到 **800%**（8 个核打满）而 GPU 还是偶尔停顿 → 用 DALI 之后 CPU 降到 **200%** 只管读文件，GPU 一边计算一边并发解码。
右框是**摆放位置的坑**：如果只用 DALI 解压 JPEG、然后把像素**交还 CPU** 做增强和 collate——多出来的 **host–device–host** 往返可能把收益全吃掉。书的替代建议：把 GPU 友好的预处理直接**融进 GPU 计算图**（TorchVision、TensorRT、自定义 CUDA kernel），端到端可能比用 DALI 更快。结论：**三种方案都测**——纯 CPU、DALI、全融合 GPU 图。

大白话：**把砧板搬到灶台边有用的前提是：食材不会中途又跑回储藏室。**

---

## 第 25 页｜Prepare Data Offline: NVIDIA NeMo Curator（离线备菜）

🎤 终极办法：**训练前把数据全部准备好**。**NeMo Curator**（开源）面向多 TB 级 LLM 数据集的离线加工：清洗、分词、shuffle、**去重**、质量过滤——而且能**分布到多 GPU/多节点**并行做。产出打包成**少量大文件**；NeMo 训练读**内存可映射的 .bin + .idx**；Curator 自己的输入输出是分片的 JSONL/Parquet。它还能造**合成数据**，补越来越稀缺的人类数据。
右框：离线备好之后，训练时几乎**只剩前向和反向**——输入一致、等长、已 padding；**不在线分词**、没有几百万小文件。
两条书里的好建议：① 存 **N 份不同 shuffle 的副本**，省掉 N 个 epoch 的运行时 shuffle——拿磁盘换速度；② 金句："**几乎永远不应该拿原始文本训练**。"注意：NeMo 的数据加载仍然走 **CPU**——要绕过 CPU I/O 得配合 **GDS**。

大白话：**头天晚上把 mise en place 全备好——开餐时只管装盘。**

---

## 第 26 页｜Part V — The Continuous Profiling and Tuning Loop（持续调优循环）

🎤 前面讲的都是一次性修复；但性能是一个**需要回归测试的 feature**。工作流三步：
① **建基线**：从单 GPU 开始量 samples/s，再扩到单机多卡、再多机——每一步看扩展性。N 块 GPU 理想拿 **N 倍**吞吐；书中例子：8 块只拿 5 倍 = **62.5% 效率**——先把差距量化出来。
② **给多卡任务做系统级 profile**（nsys）：GPU 在 **all-reduce 期间**空闲 → 通信是瓶颈；在**每次迭代开头**空闲 → 数据加载是瓶颈；同时看 CPU 时间线和同步点。
③ **必要时钻进单个 kernel**（ncu）：书中例子——一个 NCCL kernel 走 PCIe 时 SM 利用率只有 **60%**、访存 stall 高；换 NVLink 后升到 **90%**。每个 kernel 的判决必落三箱之一：**网络带宽受限、显存带宽受限、计算受限**。
右框就是循环本体：**Run → Measure → Tune → Run → Measure → Tune →…** 记住那句话：**维持**好性能比**找回**好性能容易得多。

大白话：**把性能当测试套件跑：要么今晚全过，要么明早知道是哪个 commit 弄坏的。**

---

## 第 27 页｜Identify the Cause, Fix One Thing, Remeasure（定位、修一件事、复测）

🎤 左框：把瓶颈映射到假设清单，逐个验证。**网络**：开 GPUDirect RDMA、消息攒大、多网卡时调 `NCCL_NSOCKS_PERTHREAD`、NUMA 感知绑定；**掉队 GPU**：数据不均衡，或某个 rank 在做多余的校验/日志——挪出关键路径；**CPU/数据**：加 worker、DALI 卸载、更多离线预处理；**同步**：删掉多余的 `torch.cuda.synchronize()` 和 barrier（第 13 章细讲）。
右栏四条纪律：① **一次只改一两个东西**再复测——否则不知道是谁的功劳；② **升级会挪最优点**：新版 NCCL/CUDA/PyTorch 常有免费收益——但每次升级后把 profiling 工作流**重跑一遍**；③ **自动化**：夜间 profiling、samples/s 仪表盘、利用率跌破阈值就报警；④ **把隐性知识写进代码和配置**——比如"这个集群上 `NCCL_SOCKET_NTHREADS=2` 多机吞吐 +10%"。

大白话：**像做实验一样调优：一个变量、一次测量、一条写下来的结论。**

---

## 第 28 页｜Communication- or Compute-Bound? The Batch-Size Experiment（批大小实验）

🎤 这是本章最精巧的一个诊断。场景：反向传播的梯度 all-reduce 只用到 **100 GB/s 网卡的 60 GB/s**。是网络堵，还是 GPU 慢？
关键事实：**all-reduce 的通信量只跟参数量走，跟 batch size 无关**——所以调 batch size 就是在**通信固定**的前提下调计算量。
实验一，**batch 减半**：网卡还停在 60 GB/s → GPU 本来能干更多、网络不放行——**网络受限**；网卡掉到 40 GB/s → GPU 算得不够快、喂不饱网卡——**计算受限**。
实验二，**batch 翻倍**反向验证：通信真是上限的话网卡**钉在 60**；计算是上限的话，**all-reduce 占总时间的比例会缩水**。
右框：把**实际 GB/s** 和**通信时间占比**对 batch size 画出来——这就是你系统的 roofline，天花板在哪、该调哪个子系统，一图定案。Nsight 侧的对应读法：kernel 之间长长的 **NCCL 等待**间隙 → 通信受限；GPU 忙但 FLOPS 不达预期 → 显存或计算受限（ncu / PyTorch profiler 分辨）。

大白话：**运输量按住不动、只调工作量——哪边纹丝不动，哪边就是瓶颈。**

---

## 第 29 页｜Key Takeaways (the Book's Three)（书中三条要点）

🎤 收束成三条：
① **扩计算必须同步扩输入管线**——更多 GPU 需要更多存储带宽和更多 loader 并行，否则加卡零收益；
② **用对工具**——训练的集合通信用 **NCCL**；推理的点对点 token 流用 **NIXL**；**GPUDirect RDMA / GDS** 分别为网络和存储去掉 bounce buffer；多卡训练永远 **DistributedDataParallel**，不用 DataParallel。这些库是**专门造、深度调优过的**——用它们，别重新发明轮子；
③ **端到端 profile**——瓶颈不自明：GPU 要么计算受限、要么通信受限、要么 I/O 受限；Nsight Systems + Nsight Compute + PyTorch profiler + NCCL 日志告诉你是哪个，然后本章告诉你怎么办。

大白话：**喂饱工厂、抄近道绕过前台、永远不跟人吵瓶颈在哪——量它。**

---

## 第 30 页｜Conclusion and What's Next（总结与下一步）

🎤 一句话总结（金句块）：**数据搬得快和算得快同等关键——世界上最快的 GPU，如果一直在等存储，就没什么价值。**
三点：存储是**地基层**——NVMe + GDS + 调好的管线直接缩短训练时间、**提高试验迭代速度**；你**不需要自研 I/O**——NVIDIA 和开源社区已有高度调优的专用工具，把精力留给模型、数据和应用逻辑；全栈思维继续贯穿——重叠通信与计算、用最快的链路、让每根管子都满。
下一章按书的顺序是第六章——GPU 架构、CUDA 与占用率——我们上次已经讲过了；本章的存储原则在上面每一层都持续适用。
有问题吗？

---

## 第 31 页｜References & Further Reading（参考文献）

🎤 参考文献。重点标三个：NVIDIA 的 GPUDirect Storage 文档（cuFile API 和 gdsio 用法都在里面）；DeepSeek 3FS 的开源仓库和 Fire-Flyer 论文；还有 VAST 的 GDS 基准报告——第 9 页那组 20%/30% 数字的出处。谢谢大家。
