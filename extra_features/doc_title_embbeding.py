import os
from sklearn.decomposition import PCA
# 设置 Hugging Face 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import polars as pl
from sentence_transformers import SentenceTransformer


# 加载 Sentence-BERT 模型
model = SentenceTransformer("shibing624/text2vec-base-chinese", device="cuda")


# 读取原始数据
data_path = "/data3/zxh/news_rec/raw_data"
doc_info = pl.read_ipc(f"{data_path}/doc_info.arrow")


# 计算标题的嵌入向量
titles = doc_info["title"].to_list()  # 提取标题列并转换为列表

print(len(titles))

embeddings = model.encode(titles)  # 计算嵌入

pca = PCA(n_components=32)
embeddings = pca.fit_transform(embeddings)


# 创建 DataFrame
embedding_df = pl.DataFrame(
    {
        "article_id": doc_info["article_id"],  # 保留 article_id 以便索引
        **{f"emb_{i}": embeddings[:, i] for i in range(embeddings.shape[1])}  # 按列存储嵌入
    }
)

# 保存为 IPC 格式
public_path = "/data3/zxh/news_rec/public_data"
embedding_df.write_ipc(f"{public_path}/doc_title_emb.ipc")

print(f"Title embeddings saved to: {public_path}/doc_title_emb_32.ipc")