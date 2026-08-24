# Tabero GitHub Pages 部署设计

## 目标

将现有 Tabero 静态项目页发布到独立 GitHub 仓库 `NathanWu7/Tabero`，公开地址为 `https://nathanwu7.github.io/Tabero/`，并在每次推送到 `main` 分支后通过 GitHub Actions 自动部署。

## 范围

- 保留现有 `index.html` 与 `static/` 目录结构。
- 初始化独立 Git 仓库并关联 `NathanWu7/tabero`。
- 添加 GitHub Pages Actions 工作流。
- 添加 `.nojekyll`，避免 Jekyll 处理静态资源。
- 不修改 `/home/wqw/git_pkgs/NathanWu7.github.io`。
- 根据 `NathanWu7/Tabero` README 同步真实作者、论文、代码、Hugging Face 数据资产和模型权重链接。

## 架构

仓库根目录直接作为待发布网站：

```text
tabero/
├── index.html
├── static/
├── .nojekyll
└── .github/
    └── workflows/
        └── deploy.yml
```

工作流在 `main` 分支更新或手动触发时运行：

1. 检出仓库。
2. 配置 GitHub Pages。
3. 将仓库中的静态站点上传为 Pages artifact。
4. 使用 GitHub Pages deployment API 发布。

该站点无需 Node.js、依赖安装或构建步骤。

## 路径兼容性

页面资源使用 `static/...` 相对路径。浏览器访问 `/Tabero/` 时，这些路径会解析为 `/Tabero/static/...`，无需增加 `basePath` 或改写现有 HTML。

## 权限和仓库设置

- 本地 GitHub CLI 当前未登录，创建远程仓库前需完成 `gh auth login`。
- 远程仓库应为公开仓库，以支持公开项目页面。
- Actions 工作流需要 `contents: read`、`pages: write` 和 `id-token: write` 权限。
- 仓库 Pages 来源设置为 GitHub Actions。

## 验证

- 本地启动简单静态 HTTP 服务，检查首页及主要图片、视频、PDF、CSS 和 JavaScript。
- 验证仓库内不存在超过 GitHub 单文件限制的资源；当前站点总计约 22 MB，最大文件远低于 100 MB。
- 推送后确认 Actions 成功完成。
- 检查 `https://nathanwu7.github.io/Tabero/` 返回页面，并抽查静态资源与视频播放。

## 风险与处理

- GitHub CLI 未认证：由用户完成浏览器或设备码登录后继续。
- Pages 首次启用存在短暂传播延迟：以 Actions 部署结果和最终 URL 响应为准。
- 个人主页仓库存在两个部署工作流，但本方案使用独立仓库，不会与其部署相互覆盖。
- 原页面中的匿名作者信息与 Zenodo token 将在发布前移除，并以公开的 arXiv、GitHub 和 Hugging Face 信息替换。
