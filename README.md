# rotor_order211

旋转机械阶次分析纯后端（Python 3.10.12 / FastAPI 0.115.12 / NumPy 2.2.6）。
升降速工况下振动频率随转速变化，本服务将时域振动信号按脉冲重采样到等角域，
以连续 8 转为窗、1 转为步进做阶次谱分析，返回各窗完整阶次谱（JSON）与
目标阶次幅值表（CSV 下载）。

## 模块划分

- `rotor_order211/parsing.py` — UTF-8 CSV 解析与校验（缺列、非有限值、重复/倒序
  时间、等间隔相对偏差 ≤1e-6、点数上限：振动 1e6 点 / 脉冲 1e5 个、至少 9 个脉冲）。
- `rotor_order211/anglemap.py` — 脉冲到转角映射：首脉冲为 0 转，脉冲间转角线性
  增长；每转 256 个等角点线性插值原振动；仅取连续 8 转窗口、步进 1 转，不外推、
  不拼补尾窗；脉冲必须落在振动记录时间范围内。
- `rotor_order211/spectrum.py` — 逐窗去均值、周期 Hann 窗、实数 FFT；阶次 =
  频点索引 / 8；幅值除以窗权重和，正频率翻倍，直流与奈奎斯特不翻倍，结果为峰值
  幅度（m/s²）。
- `rotor_order211/app.py` — FastAPI 交付层：`POST /analyze`（JSON 全谱）与
  `POST /analyze/csv`（目标阶次 CSV 下载）。
- `scripts/generate_sample.py` — 生成变速（600→1800 rpm 线性升速）已知阶次
  （1、2、5 阶，峰值 0.5/0.3/0.2 m/s²）样例 `data/vibration.csv` 与
  `data/pulses.csv`。

## 输入

- 振动表：`time_s,accel`（秒，m/s²），时间严格递增且等间隔。
- 脉冲表：`time_s`（秒），每脉冲一整转，与振动共用时钟。
- 目标阶次：表单字段 `orders`，逗号分隔的 1–32 整数，非空且不重复。
- 校验：最高目标阶次 × 各脉冲区间最大转频必须 < 采样率一半，否则 422。

## 输出

- JSON：单位、采样率、每窗 `rev_start/rev_end`、`time_start_s/time_end_s/
  time_mid_s`（起止时刻中点）、`avg_rpm`（= 480 / 窗口秒时长）、完整
  `orders`/`amplitudes` 数组及目标阶次幅值。
- CSV：每窗一行，含窗口范围、中点时刻、平均 RPM 与各目标阶次峰值幅值。

## 运行

```bash
.venv/bin/python scripts/generate_sample.py   # 生成样例数据
.venv/bin/python -m pytest tests -q           # 自测
.venv/bin/python -m uvicorn rotor_order211.app:app --port 8000
```

```bash
# JSON 全谱
curl -s -F vibration_file=@data/vibration.csv -F pulse_file=@data/pulses.csv \
     -F orders=1,2,5 http://127.0.0.1:8000/analyze
# CSV 下载
curl -s -F vibration_file=@data/vibration.csv -F pulse_file=@data/pulses.csv \
     -F orders=1,2,5 http://127.0.0.1:8000/analyze/csv -o orders.csv
```

## 假设与限制

- 脉冲间转角线性增长，等角重采样使用线性插值；输入信号假设相对采样率带限，
  插值不引入显著混叠。服务不做抗混叠滤波。
- 每转 256 点、8 转窗 ⇒ 阶次分辨率 1/8 阶，奈奎斯特阶次 128。
- 本服务仅做信号处理，不自动诊断故障；阶次幅值的物理解读由使用方负责。
