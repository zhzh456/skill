import spartan as st
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def eigenspokes_detect(file_path: str):
    tensor_data = st.loadTensor(path=file_path, header=None, sep=' ')
    stensor = tensor_data.toSTensor(hasvalue=False)
    es = st.Eigenspokes(stensor)
    res = es.run()
    return res

tools = [
    {
        "type": "function",
        "function": {
            "name": "eigenspokes_detect",
            "description": "Detect anomalous subgraph in graph data",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"]
            }
        }
    }
]

model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

user_input = "Detect anomaly in ./inputData/plain_graph_small.zip"

if "Detect anomaly" in user_input:
    result = eigenspokes_detect("./inputData/plain_graph_small.zip")
    print("Anomaly nodes detected:")
    print(result)