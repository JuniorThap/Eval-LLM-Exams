from openai import OpenAI

client = None
_model_name = None


def init(model_name, api_key):
    global client, _model_name
    _model_name = model_name
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key
    )

def inference(prompt):
    global client, _model_name
    completion = client.chat.completions.create(
    model=_model_name,
    messages=[{"role":"user","content":f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=True
    )

    resp = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            resp += chunk.choices[0].delta.content
    return resp