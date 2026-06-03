# HoloScope: detecting collective anoamlies of constract suspiciousness

import spartan as st
import matplotlib.pyplot as plt

# load graph data
tensor_data = st.loadTensor(path = "soc-Epinions1.csv", header=None)

# Convert to STensor - handle potential time formatting issues
# stensor = tensor_data.toSTensor(hasvalue=True, mappers={2:st.TimeMapper(timeformat='%Y-%m-%d')})
stensor = tensor_data.toSTensor(hasvalue=False)

# Create graph from STensor
graph = st.Graph(stensor, bipartite=True, weighted=True, modet=2)

# Run HoloScope as a single model
hs = st.HoloScope(graph)
print(hs)

# Run with level 0 (topology only)
res = hs.run(level=0, k=1)

# Run HoloScope from anomaly detection task
ad_model = st.AnomalyDetection.create(graph, st.ADPolicy.HoloScope, 'holoscope')

# run the model with default parameters
res = ad_model.run(k=2)

# Visualization of graphs by networkx
for r in res:
    rows, cols = r[0]
    # to subgraph
    sg = graph.get_sub_graph(rows, cols)
    # networkx plot
    fig = st.plot_graph(sg, bipartite=True, labels=[*rows, *cols])
    fig = st.plot_graph(sg, layout='circular', bipartite=True, labels=[*rows, *cols])