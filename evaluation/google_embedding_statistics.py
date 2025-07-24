import os
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly as py

# Google Embedding 001模型的向量维度
dimensions = ["768", "1536", "3072"]
metrics = ["hit_rate", "mrr"]
x_list = [f"top_{k}_retrieve" for k in range(1, 6)]

# 存储Google embedding数据
google_data = {}

# 遍历当前目录下的所有CSV文件
for file in os.listdir("."):
    if file.endswith("csv") and "gemini-embedding-001" in file:
        # 从文件名中提取向量维度
        if "gemini-embedding-001-768" in file:
            dimension = "768"
        elif "gemini-embedding-001-1536" in file:
            dimension = "1536"
        elif "gemini-embedding-001-3072" in file:
            dimension = "3072"
        else:
            continue
            
        # 读取CSV文件
        df = pd.read_csv(file)
        
        # 存储该维度的数据
        google_data[dimension] = {
            "hit_rate": [],
            "mrr": []
        }
        
        # 提取前5行的指标值
        for i in range(min(5, len(df))):
            row_data = df.iloc[i, :].to_dict()
            google_data[dimension]["hit_rate"].append(row_data["hit_rate"])
            google_data[dimension]["mrr"].append(row_data["mrr"])

# 创建子图：2行1列，分别显示hit_rate和mrr
fig = sp.make_subplots(
    rows=2, cols=1,
    subplot_titles=('Hit Rate Comparison', 'MRR Comparison'),
    vertical_spacing=0.15
)

# 定义颜色
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 蓝色、橙色、绿色

# 添加Hit Rate图
for i, dim in enumerate(dimensions):
    if dim in google_data:
        fig.add_trace(
            go.Bar(
                x=x_list[:len(google_data[dim]["hit_rate"])],
                y=google_data[dim]["hit_rate"],
                name=f"Google-001-{dim}D",
                text=[str(round(val, 4)) for val in google_data[dim]["hit_rate"]],
                textposition='auto',
                marker_color=colors[i],
                legendgroup=f"group{dim}",
                showlegend=True
            ),
            row=1, col=1
        )

# 添加MRR图
for i, dim in enumerate(dimensions):
    if dim in google_data:
        fig.add_trace(
            go.Bar(
                x=x_list[:len(google_data[dim]["mrr"])],
                y=google_data[dim]["mrr"],
                name=f"Google-001-{dim}D",
                text=[str(round(val, 4)) for val in google_data[dim]["mrr"]],
                textposition='auto',
                marker_color=colors[i],
                legendgroup=f"group{dim}",
                showlegend=False  # 避免重复图例
            ),
            row=2, col=1
        )

# 更新布局
fig.update_layout(
    title="Google Embedding 001 Models Performance Comparison",
    title_x=0.5,
    height=800,
    legend=dict(
        font=dict(size=14),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# 更新x轴和y轴标签
fig.update_xaxes(title_text="Top-K Retrieve", row=1, col=1)
fig.update_xaxes(title_text="Top-K Retrieve", row=2, col=1)
fig.update_yaxes(title_text="Hit Rate", range=[0, 1], row=1, col=1)
fig.update_yaxes(title_text="MRR Score", range=[0, 1], row=2, col=1)

# 生成HTML文件
py.offline.plot(fig, filename='google_embedding_001_comparison.html')

# 打印统计信息
print("Google Embedding 001模型性能统计完成！")
print("生成的图表文件：google_embedding_001_comparison.html")
print()

# 打印详细数据
for dim in dimensions:
    if dim in google_data:
        print(f"Google Embedding 001 - {dim}维向量:")
        print(f"  Hit Rate: {[round(val, 4) for val in google_data[dim]['hit_rate']]}")
        print(f"  MRR: {[round(val, 4) for val in google_data[dim]['mrr']]}")
        print()

# 计算并显示最佳性能
best_hit_rates = {}
best_mrr = {}

for dim in dimensions:
    if dim in google_data:
        # 使用top-2的性能作为比较基准（索引1）
        if len(google_data[dim]["hit_rate"]) > 1:
            best_hit_rates[dim] = google_data[dim]["hit_rate"][1]
            best_mrr[dim] = google_data[dim]["mrr"][1]

if best_hit_rates:
    best_hit_rate_dim = max(best_hit_rates, key=best_hit_rates.get)
    best_mrr_dim = max(best_mrr, key=best_mrr.get)
    
    print("性能总结（基于Top-2检索）:")
    print(f"  最佳Hit Rate: {best_hit_rate_dim}维 ({best_hit_rates[best_hit_rate_dim]:.4f})")
    print(f"  最佳MRR: {best_mrr_dim}维 ({best_mrr[best_mrr_dim]:.4f})") 