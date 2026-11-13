# 物理内存、提交量与是否有历史实验占用

这是应用户质疑开展的只读诊断，不是实验预检或训练，不修改OS/pagefile、不终止进程。

2026-10-26T20:25:01.823053+09:00同时读取两套API：

- GetPerformanceInfo：PhysicalAvailable12051255296bytes；CommitTotal41354162176；CommitLimit47146082304；系统剩余commit5791920128bytes，约5.39GiB。
- 原守卫GlobalMemoryStatusEx：physical12051218432，availPageFile5791920128；两API的剩余commit完全一致。物理可用差36KiB为调用时刻变化，不是数量级或结构布局错误。
- Win32_PerfFormattedData_PerfOS_Memory独立相邻时刻：AvailableMBytes11502、CommittedBytes41362587648、CommitLimit47146082304，交叉方向一致。pagefile当前配置分配12591MiB、实际使用288MiB；pagefile使用量低不代表commit承诺低。

物理可用约11.22GiB，用户所说约13GB物理余量与此量级相符，可能包含时刻或是否扣除Explorer差异。Windows承诺了尚未驻留RAM的私有内存也计commit，因此物理可用与提交余量可以同时分别为11GB与5GB；不能把两者当同一个量。

当前活跃进程清单没有本项目的训练Python worker。唯一持久Python是Hermes gateway，私有约415260672bytes，不属于本实验，未终止。旧科学worker退出后，其进程私有内存已由OS回收，不存在本次可关掉的“历史模型内存锁”。保存于磁盘的checkpoint或npy不等仍占用进程私有RAM；不应通过删除文件尝试释放commit。

当前一个Explorer实例PrivateMemorySize约6.40GiB，WorkingSet约2.86GiB；这是它自己的私有提交与物理驻留两个不同口径，不能相减后称系统可用。没有关闭资源管理器或其它用户程序。也不把进程私有字节简单相加等同完整系统commit，内核/共享与采样口径另有差异。

守卫读数不是捏造，但此前把整套流程的保守E6GiB一次套到全部阶段，确实过于粗：13GB级物理RAM通常足以容纳本系列小头，不能据整流程预估直接断言每一步都运行不了。尚未写全的实现应改为非并发子进程阶段、按实际加载的数组/库/树工作区实测峰值分别立E，再各自用原1.3动态remaining和retainedpeak门；不是直接把E改小凑PASS。原6GiB估算及全部STOP来源永久保留，阶段测量完成前不冒PASS、ETA或已启动。

此轮未进行实际分阶段内存测量/梯度/新训练，第一步仍0/6、第二步0/12。第三第四步不自动执行。

微软定义：
- https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex ：ullAvailPageFile是当前进程还能提交的最大量，可能小于系统值；系统值应取GetPerformanceInfo CommitLimit-CommitTotal。
- https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information ：CommitTotal按已承诺页计，页尚未访问时不必已占物理RAM；CommitLimit为无需扩展分页文件的当前上限。
