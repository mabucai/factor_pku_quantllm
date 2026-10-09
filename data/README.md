# 数据目录

本仓库不分发课程行情数据。复现时请将 VeighNa AlphaLab 的沪深300数据放到：

```text
data/lab/csi300/
  component/
  daily/
  contract.json
```

复制 `config.example.json` 为本地 `config.json`，将空白 `lab_path` 填为该相对路径或本机绝对路径，也可设置 `FACTOR_LAB_PATH`。`cache_dir` 留空时使用 `runs/cache`。本地配置、行情、成分股数据库和缓存均被 `.gitignore` 排除；公开样例不携带机器路径或凭据。
