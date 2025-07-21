import os

import pandas as pd
import plotly as py
import plotly.graph_objs as go

metric = "mrr"

x_list = [f"top_{k}_retrieve" for k in range(1, 6)]

# 存储各个embedding模型的MRR指标
model_mrr_dict = {}
max_metric_value = 0

# 遍历当前目录下的所有CSV文件
for file in os.listdir("."):
    if file.endswith("csv") and "embedding" in file:
        # 从文件名中提取模型名称，格式为evaluation_*_时间戳.csv
        # 提取*号部分作为模型名称
        parts = file.split("_")
        if len(parts) >= 2:
            model = parts[1]  # 提取evaluation_*_中的*部分
            
            # 初始化模型的MRR列表
            if model not in model_mrr_dict:
                model_mrr_dict[model] = []
            
            # 读取CSV文件
            df = pd.read_csv(file)
            
            # 提取前5行的MRR指标值
            for i in range(min(5, len(df))):
                metric_value = df.iloc[i, :].to_dict()[metric]
                model_mrr_dict[model].append(metric_value)
                if metric_value > max_metric_value:
                    max_metric_value = metric_value

# 按第二行（top-2）的MRR指标值进行排序
sorted_models = sorted(model_mrr_dict.items(), key=lambda x: x[1][1] if len(x[1]) > 1 else 0)

# 创建柱状图trace
trace = []
for model, metric_list in sorted_models:
    trace.append(go.Bar(x=x_list[:len(metric_list)],  # 确保x轴长度与数据长度匹配
                        y=metric_list,
                        text=[str(round(_, 4)) for _ in metric_list],
                        textposition='auto',  # 标注位置自动调整
                        name=model))

# 布局设置
layout = go.Layout(title=f'Embedding Models {metric.upper()} Comparison')

# 创建图形
figure = go.Figure(data=trace, layout=layout)
figure.add_hline(y=max_metric_value, line_width=1, line_dash="dash", line_color="red")

# 设置图例和Y轴范围
figure.update_layout(
    legend=dict(
        font=dict(
            size=16  # 设置图例文字大小
        )
    ),
    yaxis_range=[0, 1],  # MRR指标的完整取值范围0-1
    xaxis_title="Top-K Retrieve",
    yaxis_title="MRR Score"
)

# 生成HTML文件
py.offline.plot(figure, filename=f'embedding_{metric}.html')

print(f"MRR统计图表已生成：embedding_{metric}.html")
print(f"包含的模型: {list(model_mrr_dict.keys())}") 