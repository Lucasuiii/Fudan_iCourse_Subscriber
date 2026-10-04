# Qwen-only 试运行分支

此分支移除原本的本地 SenseVoice/FireRed/Zipformer 识别器，`Transcriber`
唯一运行实现是 `Qwen3-ASR-1.7B`，固定 revision `7278e1e70fe206f11671096ffdd38061171dd6e5`。
主程序、单课程手动任务和五路课程并行任务使用同一实现。未合并 main，原线上任务不变。

CPU float32 / eager / 四线程 / batch=1。VAD 仍用 sherpa-onnx 的 Silero；它不再做语音识别。
下载写入本地 PCM 后，VAD 扫描记录语音窗口，约两分钟一段，保留短停顿，跳过十秒以上的
无语音间隔；不通过音量阈值删除轻声。原媒体与 PCM 长度差异仍只警告。

词库从当前课程读取，关键词复述只重试一次，取消词库并采用60秒协作式解码预算和256 token上限。
初次识别最多2048 token；达到上限时报错，不将截断正文当作完整识别存入数据库。
纯语气词结果不进入正文，其他短技术词保留。每节课后释放模型，给OCR及强制对齐留内存。

豆包最多10分钟/节、12段。Qwen有明确原文疑点时使用完整两分钟块的强制对齐，
定位失败不上传。不直接把包含上下文的豆包片段覆盖到原句；原句与复核版本一并供摘要判断。
官方字幕保持辅助完整性检查与保守补空，不升为主要转写来源。

新实现的模型推理仍由Actions作业总时限兜底，逐块总时限检查不是强制中断原生计算。
目前没有跨Actions运行的ASR断点恢复；失败课次下次会重跑，不会回退SenseVoice。
旧数据库中的已完成摘要保留，不因识别引擎切换自动重算历史课程。

Actions明确安装CPU版 `torch==2.11.0` 与 `qwen-asr==0.0.6 --no-deps`，其余依赖在requirements。
本地使用也须额外安装这两个包，不应让pip自动安装CUDA版torch。
模型及强制对齐权重缓存于Hugging Face hub缓存；不再下载SenseVoice模型。

基准工作流可用 `runtime_sample=true, public_sample=true`，运行官方公开音频验证新主程序入口。
该模式不登录iCourse、不调用摘要/豆包、不写正式数据库、不发邮件。

完整课堂隔离测试可设置 `runtime_sample=true, full_lecture=true`，调用同一生产
`Transcriber`，保留原始块、VAD 窗口及失败诊断供复核；到达三小时获取上限不标为完成。
这验证真实识别入口与隔离复核/摘要，不等同于完整 `LectureRunner`、邮件和数据库发布验证。
`course_slots=pair` 可让两个独立 Runner 并行，选课来自专用
`QWEN_ASR_TEST_REQUESTS` Secret 的两个条目，日志和产物名称只显示序号。
该模式要求 `latest_lecture=true`，按日期逐次解析既有播放回退链，选最新可播放的非未来录播；
不依赖 `has_playback` 标记，被跳过的无地址课次及实际选课日期保留在加密产物中。
所有非未来课次都无地址才报错。正式 `COURSE_IDS` 和原单课测试 Secret 不变。
`course_slots=0` 或 `1` 可单独重试失败的一门，避免重跑已成功的另一门。
同一批次内每个课程序号使用独立并发组，同一序号排队，不同序号可并发。

单堂课跨 Runner 试验入口见 [分片试验说明](qwen-sharded-pilot.md)。
新提交的基准测试与分片／订阅入口共用批次并发组；已经在旧提交上运行的任务不受影响。
