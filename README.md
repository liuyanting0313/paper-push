# arXiv 每日论文摘要推送

每天定时从 arXiv 按你设置的分类 + 关键词拉取最新论文，整理成摘要邮件发送给自己。

## 功能特点

- 按 arXiv 分类(如 `cs.RO`、`eess.SY`、`cs.SY`、`cs.AI`、`cs.LG`)和关键词组合过滤论文
- 支持排除关键词(`exclude_keywords`): 命中即直接剔除, 避免噱头论文混入
- 相关度排序: 按关键词命中次数(标题命中额外加权)计算得分, 支持 `keyword_weights` 自定义权重, 高相关论文排在最前
- 自动去重: 记录已推送过的论文 ID(`sent_ids.json`), 默认忽略版本号(v1/v2 视为同一篇), 不会重复推送
- 邮件采用学术日报排版, 每篇论文包含:
  - 标题超链接(直达 arXiv 详情页)、PDF 直达链接、arXiv 编号
  - 完整作者名单、首发日期与最新更新日期(如有修订版本)
  - 主分类 + 次分类标签(带人类可读名称, 如 `cs.RO (Robotics)`)
  - 命中关键词高亮标签 + 相关度得分, 并在摘要正文中用 `<mark>` 标出命中片段
  - 完整英文 Abstract(不截断), 采用衬线字体+两端对齐, 阅读体验接近论文排版
  - 中文摘要机器翻译, 或(启用 LLM 后)AI 生成的创新点/方法/结论中文精读要点
  - 可折叠的 BibTeX 引用代码块, 方便写论文时直接复制
- 邮件同时发送 HTML + 纯文本两个版本, 部分客户端屏蔽 HTML 时也能看到内容; 支持多个收件人(逗号分隔或列表)
- 可选同时推送到企业微信 / 飞书 / Slack 的 Webhook 机器人(`push` 配置项)
- 网络请求(拉取论文/翻译/LLM调用/邮件发送/Webhook)均带指数退避重试, 提升弱网环境下的稳定性
- 支持 `--dry-run` 模式先在控制台/日志里预览完整结果, 不发送邮件
- 运行日志记录到 `logs/arxiv_digest.log`, 方便排查问题

## 目录结构

```
new-task/
├── arxiv_digest.py         # 主脚本
├── config.example.yaml     # 配置文件模板(可提交到 git)
├── config.yaml             # 你的真实配置(包含密码, 已被 .gitignore 忽略)
├── requirements.txt        # Python 依赖
├── sent_ids.json           # 已推送论文记录(运行后自动生成, 已忽略)
└── logs/                   # 运行日志(自动生成, 已忽略)
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

复制配置模板并编辑:

```bash
copy config.example.yaml config.yaml
```

打开 `config.yaml`, 至少需要修改:

- `arxiv.categories` / `arxiv.keywords`: 你关心的分类和关键词
- `email.sender`: 你的 QQ 邮箱地址
- `email.password`: QQ 邮箱的 **SMTP 授权码**(不是登录密码!)
  获取方式: QQ邮箱网页版 → 设置 → 账户 → POP3/IMAP/SMTP服务 → 开启服务并生成授权码
- `email.receiver`: 接收摘要的邮箱地址(可以填自己)

### 3. 先用 dry-run 验证配置

不会真正发邮件, 只在控制台/日志打印筛选出的论文:

```bash
python arxiv_digest.py --dry-run
```

### 4. 正式运行(发送邮件)

```bash
python arxiv_digest.py
```

首次正式运行成功后, 推送过的论文 ID 会写入 `sent_ids.json`, 之后再次运行不会重复推送同一篇论文。

## 设置每天定时运行(Windows 任务计划程序)

用管理员权限打开 PowerShell, 执行(请把路径替换成你实际的 Python 和脚本路径):

```powershell
schtasks /create /tn "arXiv每日论文摘要" /tr "python D:\software\Antigravity\Agent_Workspace\new-task\arxiv_digest.py" /sc daily /st 08:00
```

- `/sc daily /st 08:00` 表示每天早上 8:00 执行, 可按需修改时间
- 创建后可以在"任务计划程序"图形界面里找到该任务, 手动运行一次测试
- 删除任务: `schtasks /delete /tn "arXiv每日论文摘要" /f`

也可以用图形界面操作: 打开"任务计划程序" → 创建基本任务 → 触发器选"每天" → 操作选择"启动程序", 程序填 `python.exe` 的完整路径, 参数填 `arxiv_digest.py` 的完整路径, 起始于填脚本所在目录。

## 常见问题

- **邮件发送失败 / 认证错误**: 检查 `email.password` 是否填的是 SMTP 授权码而不是 QQ 登录密码, 以及 `email.sender` 邮箱是否已开启 SMTP 服务。
- **没有筛选到任何论文**: 尝试调大 `arxiv.days_back`, 或放宽关键词; `categories` 和 `keywords` 是 AND 关系, 两者都很严格时容易查不到结果。
- **想同时按分类或关键词二选一过滤**: 把其中一项设为空列表 `[]`。
- **改了关键词后想重新收到之前发过的论文**: 手动清空或编辑 `sent_ids.json`。
