# GTCC Online

联合循环机组在线性能计算程序。输入 JSON，输出性能计算结果 JSON。

## 安装环境

```powershell
uv sync
```

## 运行

```powershell
uv run python energy_analysis\PLANT_model.py\plant_model.py input\example\input.json
```

传入自己的 JSON 文件：

```powershell
uv run python energy_analysis\PLANT_model.py\plant_model.py C:\path\to\input.json

uv run python energy_analysis\PLANT_model.py\plant_model.py testdata\real_sample_001.json
```

## 常用参数

```powershell
--frame-index 0
--output output\output.json
```

完整示例：

```powershell
uv run python energy_analysis\PLANT_model.py\plant_model.py C:\path\to\input.json --output output\output.json
```

## 输入输出

输入 JSON 需要包含：

- `point_table`：测点编码列表
- `frames`：数据帧列表

默认输出：

- `output\output.json`

如需输出字段说明文件，可额外添加：

```powershell
--comments-output output\output_mapping_comments.json
```
