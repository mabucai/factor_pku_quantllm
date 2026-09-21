# 数据目录

本仓库不分发课程行情数据。复现时请将 VeighNa AlphaLab 的沪深300数据放到：

```text
data/lab/csi300/
  component/
  daily/
  contract.json
```

`config.json` 中的 `lab_path` 已配置为该相对路径。行情、成分股数据库和缓存均被 `.gitignore` 排除。
