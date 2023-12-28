import os

import pandas as pd
import plotly as py
import plotly.graph_objs as go

metric = "hit_rate"

x_list = [f"top_{k}_retrieve" for k in range(1, 6)]

model_hit_rate_dict = {"embedding": [],
                       "bm25": [],
                       "ensemble": [],
                       "ensemble-rerank": []}

max_metric_value = 0

for file in os.listdir("."):
    if file.endswith("csv"):
        model = file.split("_")[1]
        df = pd.read_csv(file)
        for i in range(5):
            try:
                metric_value = df.iloc[i, :].to_dict()[metric]
                model_hit_rate_dict[model].append(metric_value)
                if metric_value > max_metric_value:
                    max_metric_value = metric_value
            except Exception:
                model_hit_rate_dict[model].append(0)

trace = []
for model, metric_list in model_hit_rate_dict.items():
    trace.append(go.Bar(x=x_list,
                        y=metric_list,
                        text=[str(round(_, 4)) for _ in metric_list],
                        textposition='auto',  # 标注位置自动调整
                        name=model))

# Layout
layout = go.Layout(title=f'Retrieve {metric} experiment')
# Figure
figure = go.Figure(data=trace, layout=layout)
figure.add_hline(y=max_metric_value, line_width=1, line_dash="dash", line_color="red")
# Plot
py.offline.plot(figure, filename=f'{metric}.html')
