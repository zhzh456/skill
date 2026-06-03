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
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

user_input = "Detect anomaly in ./inputData/plain_graph_small.zip"

prompt = f"You can use tools: {json.dumps(tools)}\nUser: {user_input}\nOutput function call in JSON format: "

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)

result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
tool_call = json.loads(result_text)

if tool_call["name"] == "eigenspokes_detect":
    file_path = tool_call["parameters"]["file_path"]
    res = eigenspokes_detect(file_path)
    print("Anomaly nodes detected:")
    print(res)