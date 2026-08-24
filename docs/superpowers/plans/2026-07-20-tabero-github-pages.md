# Tabero GitHub Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Tabero 静态网站从独立仓库自动发布到 `https://nathanwu7.github.io/Tabero/`。

**Architecture:** 现有 Tabero 代码仓库的根目录同时保存无需构建的静态项目页。GitHub Actions 仅将 `index.html`、`.nojekyll` 和 `static/` 复制到临时 `_site` 目录并打包为 Pages artifact，避免发布仓库源码。

**Tech Stack:** HTML/CSS/JavaScript、Git、GitHub Actions、GitHub Pages

---

## 文件结构

- 创建 `.github/workflows/deploy.yml`：定义静态站点的 Pages 自动部署流程。
- 创建 `.nojekyll`：禁止 GitHub Pages 对静态资源执行 Jekyll 处理。
- 修改 `index.html`：网站入口，同步公开作者及项目资源链接。
- 修改 `README.md`：将过期的 Zenodo badge 替换为 Hugging Face 数据集入口。
- 保留 `static/`：图片、视频、PDF、CSS 和 JavaScript，不修改内容。

### Task 1: 添加 GitHub Pages 工作流

**Files:**
- Create: `.github/workflows/deploy.yml`
- Create: `.nojekyll`

- [ ] **Step 1: 创建部署工作流**

写入：

```yaml
name: Deploy Tabero to GitHub Pages

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          lfs: true
      - name: Configure Pages
        uses: actions/configure-pages@v5
      - name: Prepare static site
        run: |
          mkdir _site
          cp index.html .nojekyll _site/
          cp -r static _site/
      - name: Upload static site
        uses: actions/upload-pages-artifact@v3
        with:
          path: _site
      - name: Deploy
        id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: 创建 `.nojekyll`**

Run: `touch .nojekyll`

Expected: `.nojekyll` 存在且为空。

- [ ] **Step 3: 校验工作流 YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/deploy.yml')); print('YAML OK')"`

Expected: 输出 `YAML OK`。

### Task 2: 本地验证静态站点

**Files:**
- Verify: `index.html`
- Verify: `static/`

- [ ] **Step 1: 检查 HTML 引用的本地资源是否存在**

Run:

```bash
python - <<'PY'
from pathlib import Path
import re

html = Path("index.html").read_text()
paths = sorted(set(re.findall(r'(?:src|href)="(static/[^"]+)"', html)))
missing = [path for path in paths if not Path(path).is_file()]
print(f"checked={len(paths)} missing={len(missing)}")
if missing:
    print("\n".join(missing))
    raise SystemExit(1)
PY
```

Expected: `missing=0`。

- [ ] **Step 2: 启动本地静态服务器并检查入口**

Run: `python -m http.server 8000`

另一个终端运行：`curl --fail http://127.0.0.1:8000/`

Expected: HTTP 请求成功并返回 Tabero HTML。

- [ ] **Step 3: 检查仓库大小限制**

Run: `du -sh . && du -ah static | sort -h`

Expected: 总大小约 22 MB，所有单文件均小于 GitHub 的 100 MB 限制。

### Task 3: 同步公开项目链接

**Files:**
- Modify: `index.html`
- Modify: `README.md`

- [ ] **Step 1: 同步网站作者与资源入口**

将论文入口改为 `https://arxiv.org/abs/2605.27886`，代码入口改为 `https://github.com/NathanWu7/Tabero`，并添加 README 中列出的 Assets、Tactile Assets 和 Model Weights 三个 Hugging Face 按钮。使用 README Citation 中的真实作者和 BibTeX 替换匿名信息。

- [ ] **Step 2: 删除 Zenodo token**

Run: `rg 'Anonymous|anonymous\.4open|openreview\.net|zenodo\.org|token=' index.html`

Expected: 无匹配结果。

- [ ] **Step 3: 更新 README 数据集 badge**

将 README 顶部 Zenodo badge 改为指向 `https://huggingface.co/datasets/NathanWu7/Isaaclab_Libero` 的 Hugging Face badge。

### Task 4: 初始化并发布独立仓库

**Files:**
- Track: `index.html`
- Track: `static/`
- Track: `.github/workflows/deploy.yml`
- Track: `.nojekyll`
- Track: `docs/superpowers/`

- [ ] **Step 1: 验证 GitHub SSH 与 CLI 认证状态**

Run: `ssh -T git@github.com`

Expected: GitHub 返回已成功认证的账号信息；该命令可能以状态码 1 结束，这是 GitHub SSH 探测的正常行为。

Run: `gh auth status`

Expected: 显示已登录 `NathanWu7`。如果 CLI 尚未认证，运行 `gh auth login --web --git-protocol ssh` 并由用户完成设备授权。

- [ ] **Step 2: 初始化 Git 仓库**

Run: `git init -b main`

Expected: 当前目录成为 `main` 分支的 Git 仓库。

- [ ] **Step 3: 检查待提交内容**

Run: `git status --short && git diff --no-index /dev/null .github/workflows/deploy.yml`

Expected: 网站、工作流和文档均为未跟踪文件，工作流内容与 Task 1 一致。

- [ ] **Step 4: 创建首次提交**

Run:

```bash
git add . &&
git commit -m "feat: publish Tabero project website"
```

Expected: 创建包含完整静态站点及部署配置的首次提交。

- [ ] **Step 5: 创建公开仓库并推送**

Run: `gh repo create NathanWu7/tabero --public --source=. --remote=origin --push`

Expected: 创建 `https://github.com/NathanWu7/tabero` 并推送 `main`。

### Task 5: 验证线上部署

**Files:**
- Verify: `.github/workflows/deploy.yml`

- [ ] **Step 1: 查看部署运行**

Run: `gh run list --workflow deploy.yml --limit 1`

Expected: 最新运行最终状态为 `completed success`。

- [ ] **Step 2: 检查 Pages 地址**

Run: `curl --fail --retry 6 --retry-delay 10 https://nathanwu7.github.io/Tabero/`

Expected: 返回 Tabero 页面 HTML。

- [ ] **Step 3: 抽查主要静态资源**

Run:

```bash
curl --fail --head https://nathanwu7.github.io/Tabero/static/css/index.css &&
curl --fail --head https://nathanwu7.github.io/Tabero/static/videos/task1_gentle_success.mp4
```

Expected: 两个请求均返回成功状态。
