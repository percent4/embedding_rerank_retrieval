# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: late_chunking_exp.py
# @time: 2024/12/22 22:48
from transformers import AutoModel
from transformers import AutoTokenizer
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True)
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True)

chunks = [
    "蔚来ET9正式上市 售78.8万元起",
    "易车讯 12月21日，蔚来ET9正式上市，售价区间78.8-81.8万元。蔚来ET9定位蔚来品牌科技行政旗舰轿车，新车搭载众多顶尖的黑科技，是中国首款搭载线控转向技术的量产车型，并搭载先进数字架构。",
    "蔚来ET9保留了蔚来家族式设计，标志性的一体式X-Bar和Double Dash日间行车灯，让新车看起来富有力量感。“Design for AD”的设计理念得以延续，前瞭望塔式传感器布局，将3颗激光雷达、摄像头等感应硬件巧妙融入外观造型设计中。",
    "车头大灯组采用了行业首发MicroLED智能高像素大灯，结合Aqulia2.0超感系统可以实现“广、亮、准、远”的精细化照明。新车整体造型非常流畅，车顶流线从车头一直延伸到车尾，像一张巨大的弓箭，在保持了经典轿车造型商务感的同时，又带来强大的气场和未来气息。",
    "超感系统天鹰座Aquila 2.0新增双侧广角激光雷达，通过两侧金属翼子板集成，即提升了安全性，又提升了辅助驾驶能力。超远距激光雷达，搭载蔚来自研杨戬主控芯片，成像效果更佳清晰。新车首次搭载4D毫米波成像雷达，大大增加前向感知能力。",
    "车身尺寸方面，蔚来ET9长宽高分别为5325*2017*1621mm，轴距达到了3250mm。此外，新车还配备了23寸的铝合金锻造轮毂，且搭配同级最大的790mm轮胎直径，极具视觉冲击力。来到车尾，新车延续了家族式设计，贯穿式的尾灯组极具辨识度。值得一提的是，新车搭配了同级唯一的鹅颈式全主动尾翼，运动感十足。蔚来ET9首发感应式电动前备箱，支持脚踢感应和车外语音开启，前备箱容积达到105L。",
    "内饰方面，蔚来ET9首次采用了矩形方向盘，同时，新车还首发搭载蓝宝石全焦段 AR HUD，能够实现远焦面15米处等效120寸AR-HUD效果。",
    "作为行政旗舰轿车，蔚来ET9采用四座布局，创造性的采用了“天空岛”和“行政桥”的设计，配合拱式车身设计，后排的乘坐体验堪比商务舱。在'行政桥'内部，蔚来为二排乘客精心设计了飞机头等舱座椅，拥有582mm超宽坐垫，拥有前排22向，后排20向电动调节。此外，二排座椅还拥有135°超大躺角，可一键尊享11项功能联动。后排还配备了一张360°无级调节的行政桌案，能在任意角度随心调节。“行政桥”下方集成智能冰箱，最大容积达到10L，温度调节范围在-2°C到55°C，此外还首发了常温模式，总计拥有6种预设模式。",
    "此外，全车配备七扇电动遮阳帘，支持一键开启。专为后排商务场景开发的全景互联行政屏，应用14.5英寸OLED高清显示屏，屏幕角度可随座椅位置调节，任意姿态下都能拥有舒适的视角。",
    "蔚来ET9还首发九霄天琴蔚来8.2.4.8旗舰沉浸声系统。配备了35个扬声器，采用8.2.4.8声学布局，功率可达2800W。在ET9后排的行政桥内，还设置了中置环绕单元，配备了2个高音扬声器+1个中音扬声器。",
    "蔚来ET9还首发搭载cedar 雪松全新智能系统，包含全新一代感知硬件、全新一代中央计算器、SkyOS 天枢整车操作系统等。ET9搭载了蔚来首款5nm车规级智能驾驶芯片——神玑NX9031，与全球首个以车为中心的智能电动汽车整车全域操作系统SkyOS·天枢相结合，实现算力与自身算法的紧密结合，智驾、座舱跨域计算资源的共享，带来极致安全和极致效率。",
    "蔚来ET9搭载先进数字架构，定义了一层解耦的计算与通信框架，能够支持智能硬件、操作系统、算法和应用等各层次独立迭代。具体来看，蔚来ET9的先进数字架构由大脑——中央计算平台、小脑与脊髓——高效区域控制器、神经网络——高速冗余的通信网络、血液循环——双冗余低压电源、感知器官——超感系统、灵魂和思想——整车全域操作系统六大部分组成，具备强大的算力、超高带宽与极低时延、极致可靠、精准到点的能源管理等特点。在先进数字架构的支撑下，蔚来ET9实现了多项全球首发与同级领先的技术。",
    "SkyOS是蔚来整车底层操作系统，包含了整车系统、智驾系统、智能座舱系统、联通服务补能和移动互联，解决整车各个系统不同域之间的安全性、实时性和应用的复杂性问题，以及将软件定义汽车有效落实到造车的各个环节，建立全方位的、立体的技术体系，使得各种设备能够有机地融合在一起，实现高效的协同工作。",
    "蔚来ET9搭载国内首个“全域900V高压架构”，包含电池、电机、线束、空调、DC-DC、车载充电机等核心电子电器元件，拥有最高电压925V、充电峰值功率600kW、充电峰值电流765A的三项全球第一。",
    "具体来看，蔚来ET9搭载了前180千瓦感应异步电机，后340千瓦永磁同步电机，综合功率520千瓦，综合扭矩达700牛·米，百公里加速4.3秒。电池方面，蔚来ET9搭载自研46105大圆柱电芯。补能方面，新车的闪电充峰值功率高达600kW，充电峰值电流765A，900V支持充电5分钟补能255公里。",
    "蔚来ET9搭载“SkyRide·天行智能底盘系统”，首次将线控转向、后轮转向和全主动悬架三大核心硬件系统集成在一起，是目前全球唯一的全线控智能底盘。全球首创智能化、高集成度的主动悬架系统，每个减振器高度集成独立电动液压泵，无刷直流电机响应迅速，可以在1毫秒内完成信息处理、计算和响应。同时，悬架支持大幅度高度调节，每秒可进行1000次扭矩调整，且四轮独立控制，满足多场景驾驶需求。",
    "蔚来ET9首次应用的航空工业级“线控转向”技术，方向盘与转向电机之间采用电讯号传输，不仅结构重量轻，传递效率也能提升40%，并支持OTA迭代升级。在低速泊车、掉头等场景中，“线控转向”技术提供灵敏便捷的转向，无需交叉手打方向盘，配合标配最大后轮转角8.3°的后轮转向系统，实现最小10.9米的转弯直径。",
    "天行全主动悬架的每个减振器高度集成独立电动液压泵，无刷直流电机响应迅速，可以在1毫秒内完成信息处理、计算和响应。同时，悬架支持大幅度高度调节，每秒可进行1000次扭矩调整，且四轮独立控制，满足多场景驾驶需求。",
    "车身强度方面，新车采用高强度钢铝镁合金车身与空间力学设计，扭转刚度达52600Nm/Deg。车身强度达2000MPa，全面提升乘员舱保护。侧气帘长2.3m，高0.67m，可100%覆盖前后排乘客保护区域。同时，新车搭载了行业首创“V腔”设计的二排专属侧气囊。"
]

input_text = ''.join(chunks)

chunk_inputs = tokenizer(chunks[0], return_tensors='pt')
first_length = chunk_inputs['input_ids'].shape[1]
span_annotations = [(1, first_length)]

for i in range(1, len(chunks)):
    chunk_inputs = tokenizer(chunks[i], return_tensors='pt')
    length = chunk_inputs['input_ids'].shape[1]
    start = span_annotations[i-1][1]
    end = start + length
    span_annotations.append((start, end))

print(span_annotations)

def late_chunking(
    model_output: 'BatchEncoding', span_annotation: list, max_length=None
):
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if (
            max_length is not None
        ):  # remove annotations which go bejond the max-length of the model
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)

    return outputs

# chunk before
embeddings_traditional_chunking = model.encode(chunks)

# chunk after wards (context-sensitive chunked pooling)
inputs = tokenizer(input_text, return_tensors='pt', max_length=4096, truncation=True)
model_output = model(**inputs)
embeddings = late_chunking(model_output, [span_annotations])[0]

cos_sim = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

query = "蔚来ET9中的冰箱的最大容积是多少？"
query_embedding = model.encode(query)

naive_embedding_score_dict = {}
late_chunking_embedding_score_dict = {}
for chunk, trad_embed, new_embed in zip(chunks, embeddings_traditional_chunking, embeddings):
    # 计算query和每个chunk的embedding的cosine similarity，相似度分数转化为float类型
    naive_embedding_score_dict[chunk] = cos_sim(query_embedding, trad_embed)
    late_chunking_embedding_score_dict[chunk] = cos_sim(query_embedding, new_embed)

naive_embedding_order = sorted(
    naive_embedding_score_dict.items(), key=lambda x: x[1], reverse=True
)
late_chunking_order = sorted(
    late_chunking_embedding_score_dict.items(), key=lambda x: x[1], reverse=True
)


def get_answer(query, retrieve_result):
    top_k = 4
    text = ''.join([_[0] for _ in retrieve_result[:top_k]])
    prompt = f"给定下面的文本，请问答用户的问题。\n\n{text}\n\n问题：{query}"

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content


naive_embedding_answer = get_answer(query=query, retrieve_result=naive_embedding_order)
print(f"query: {query}, 朴素嵌入时RAG过程中LLM的回复：{naive_embedding_answer}")
late_chunking_answer = get_answer(query=query, retrieve_result=late_chunking_order)
print(f"query: {query}, 迟分嵌入时RAG过程中LLM的回复：{late_chunking_answer}")
