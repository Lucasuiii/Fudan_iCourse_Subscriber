# 个人部署说明（GitHub Actions）

本项目默认运行在 GitHub 提供的临时 Ubuntu runner 上，不需要让本地电脑持续
开机。仓库是公开 Fork，因此 Actions 日志和 `data` 分支中的加密文件也是公开
可访问的；任何凭证都只能存入 GitHub Actions Secrets，不能写入代码或提交。

## 部署前边界

- 仅处理本人有权访问的课程资料。
- GitHub Pages 前端仅在当前标签页的 `sessionStorage` 中保存 PAT 和
  `DB_ENCRYPTION_KEY`，关闭标签页后失效；不要在公共电脑上使用。
- 前端 PAT 只授予当前仓库的 Actions/Secrets 读写和 Contents 只读权限。
- 不公开或转发录播、转录、PPT OCR 与课程摘要。
- 上游更新不会自动进入本 Fork；合并前应人工审查网络请求和 workflow 变更。

## 需要配置的 Secrets

进入自己的仓库：`Settings -> Secrets and variables -> Actions`，添加：

| Secret | 内容 |
|---|---|
| `STUID` | 复旦学号 |
| `UISPSW` | UIS 密码 |
| `COURSE_IDS` | 每日订阅课程 ID，多个用英文逗号分隔 |
| `COURSE_SESSION_RULES` | 可选；每行一门课程的课次白名单，例如 `35472=周一第1-2节|周三第6-8节` |
| `COURSE_SESSION_OVERRIDE_DATES` | 可选；调课/补课日期例外，逗号分隔，例如 `2026-09-20,2026-10-01` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key；首次只配置这一个模型服务即可 |
| `TAVILY_API_KEY` | 可选。仅当笔记标出可公开核查的术语缺口时使用；每节课最多 2 次基础搜索，不上传整段课堂材料。未配置时不联网检索。 |
| `SMTP_EMAIL` | QQ 发件邮箱 |
| `SMTP_PASSWORD` | QQ 邮箱 SMTP 授权码，不是邮箱登录密码 |
| `RECEIVER_EMAILS` | 接收摘要的邮箱；多个地址用英文逗号、分号或换行分隔 |
| `RECEIVER_EMAIL` | 兼容旧部署；未设置 `RECEIVER_EMAILS` 时使用的单个邮箱 |
| `DB_ENCRYPTION_KEY` | 独立随机数据库密钥，不能与 UIS 密码相同 |

在本地终端生成数据库密钥：

```bash
openssl rand -hex 32
```

只把输出粘贴到 `DB_ENCRYPTION_KEY` Secret。不要把输出发给别人，也不要写入
`.env` 后提交。丢失该密钥将无法解密已有数据库；更换它之前应先做好迁移。

Fork 的 Actions 如处于禁用状态，进入 `Actions` 页面，阅读提示后为该 Fork 启用
workflows。先不要运行任何 workflow，等下面的 Secrets 全部配置完成。

`COURSE_SESSION_RULES` 支持每行一个课程。没有出现在该 Secret 中的课程会处理全部
可播放课次；出现的课程只处理列出的星期和节次。多条规则使用 `|` 分隔，也可填写
`课程ID=全部`。格式错误时任务会在登录和调用模型前停止，且公开日志不会打印规则
内容。

若临时调课落在白名单之外，可把实际上课日期加入
`COURSE_SESSION_OVERRIDE_DATES`。该日期会对所有已配置课程放行；课程处理成功后保留
这个日期也不会重复生成摘要。

每日任务优先使用完整的 iCourse 官方字幕，并用实际媒体时长检查头尾覆盖。官方字幕
只有少量超过 20 分钟的缺口时，仅对缺口运行本地 ASR 并按时间轴合并；字幕缺失、
过于稀疏或大部分内容缺失时才进行完整 ASR。每门课程会单独发送一封邮件，正文保留
HTML 预览，并将相同 HTML 渲染为 PDF 附件。PDF 生成失败时会自动改附 `.md`
文件，邮件本身不会因此丢失。

每日自动任务在北京时间 17:07 主运行。20:07 会先检查当天是否已有成功、排队中或
运行中的定时任务；只有 17:07 那轮仍未开始或已经失败时，才执行完整保底任务。
已处理课次会由数据库自动跳过；GitHub cron 可能因队列繁忙而延迟，不能作为精确到
分钟的定时器。

同一课次连续失败三次后会暂停自动重试，并向 `RECEIVER_EMAILS` 发送一次私人失败
摘要；摘要只包含课程、课次、失败阶段和次数，不包含异常原文、签名 URL 或凭证。
提醒发送失败时不会标记为已通知，下次任务会继续尝试发送。需要重新处理时，打开
`Actions -> Single Run -> Run workflow`，勾选
`Retry all paused failed lectures`。这是布尔开关，不需要在公开的 workflow 参数里
填写课程或课次 ID。

## 首次试跑

1. 在 `COURSE_IDS` 中暂时只填一门课程。
2. 打开 `Actions -> Single Run -> Run workflow`。
3. “优先使用官方字幕”现在默认开启；课程 ID 会直接从 `COURSE_IDS` Secret 读取，
   不在公开参数中填写。
4. 运行后检查 Actions 日志中没有课程名、教师名、课次 ID、完整 URL 或邮箱地址。
5. 检查邮件和模型平台账单；确认无异常后，再把其他课程加入 `COURSE_IDS`。

首次运行默认会处理所选课程的所有已有录播；设置 `COURSE_SESSION_RULES` 后，只处理
符合白名单的课次。视频地址获取和开放时间处理保持不变；失败课次按上述规则最多
自动尝试三次。GitHub 的 cron 不保证准点执行。

官方字幕缺失或不完整时，SenseVoice ASR 在 GitHub Actions runner 本地运行，不需要
额外的语音识别 API。课程标题、转录/OCR 文本和摘要提示会发送给你配置的模型服务商；生成的摘要会发送
给邮箱服务商。请按课程资料的使用规则和对应服务商的隐私条款决定是否使用。

## 可选：导出或删除数据

在前端课程页可按课次导出或删除。课次选择会先写入一次性 GitHub Secret，公开的
workflow 参数只包含随机请求 ID。删除会清除摘要、转录和 PPT OCR，但保留一个不含
课程内容的永久忽略标记，因此后续每日任务不会重新生成或补发该课次。

## 停用

先在仓库 `Actions` 中禁用 workflows，再撤销模型 API Key 和 SMTP 授权码。若
出现 UIS 异地登录或异常认证重试，应立即停用 workflow 并更改 UIS 密码。
