# FlowScope: Spotting Money Laundering Based on Graphs

import spartan as st
import matplotlib.pyplot as plt
import numpy as np

# load graph data
fs1_tensor_data = st.loadTensor(path = "./inputData/fs_in_data.csv.gz", header=None)
fs2_tensor_data = st.loadTensor(path = "./inputData/fs_out_data.csv.gz", header=None)

fs1_stensor = fs1_tensor_data.toSTensor(hasvalue=True)
fs2_stensor = fs2_tensor_data.toSTensor(hasvalue=True)

# Change the shape of two stensors, make sure they have the same size in middle dimension
maxshape = max(fs1_stensor.shape[1], fs2_stensor.shape[0])
fs1_stensor.shape = (fs1_stensor.shape[0], maxshape)
fs2_stensor.shape = (maxshape, fs2_stensor.shape[1])

graph_1 = st.Graph(fs1_stensor, bipartite=True, weighted=True, modet=None)
graph_2 = st.Graph(fs2_stensor, bipartite=True, weighted=True, modet=None)

# Create a graph list, and add graphs in order
step2list = []
step2list.append(graph_1)
step2list.append(graph_2)

# Run FlowScope as a single model
fs = st.FlowScope(step2list)
print(fs)

# Default parameters are: {'alpha': '4'}
# alpha is equivalent to $\lambda$ in the paper
res = fs.run(k=3, alpha=4, maxsize=(10,10,10))

# Run FlowScope from anomaly detection task
ad_model = st.AnomalyDetection.create(step2list, st.ADPolicy.FlowScope, 'flowscope')

# run the model
# default k=3, alpha=4
res = ad_model.run(k=3, alpha=4, maxsize=(-1,-1,100))

# Visualization of graphs by networkx
for r in res:
    one, two, three = r[0]
    one = np.array(one)
    two = np.array(two)
    three = np.array(three)
    # to subgraph
    sg_1 = graph_1.get_sub_graph(one, two)
    sg_2 = graph_2.get_sub_graph(two, three)
    # networkx plot
    fig_1 = st.plot_graph(sg_1, bipartite=True, labels=[*one, *two])
    fig_2 = st.plot_graph(sg_2, bipartite=True, labels=[*two, *three])