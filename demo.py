import spartan as st
import pandas as pd

# 你的原有代码
tensor_data = st.loadTensor(path="./inputData/plain_graph_small.zip", header=None, sep=' ')

# ======================================
# 新增：把 tensor_data.data 导出为 CSV 文件
# ======================================
# 转成 pandas DataFrame
df = pd.DataFrame(tensor_data.data)  # 关键：取 .data 才能拿到真实数组

# 保存为 CSV（无索引、无表头，和原图数据格式一致）
df.to_csv("./tensor_data_export.csv", index=False, header=False)
print("✅ 已成功保存为 CSV：./tensor_data_export.csv")

# 你的后续代码
stensor = tensor_data.toSTensor(hasvalue=False)
es = st.Eigenspokes(stensor)
res = es.run()