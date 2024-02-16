import time
from ctransformers import AutoModelForCausalLM

"""
1. Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
2. Set the system message
"""
config = {
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
    "temperature": 0.1,
    "stream": False,
}

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGML",
    model_file="llama-2-7b-chat.ggmlv3.q4_K_M.bin",
    model_type="llama",
    gpu_layers=20,
    **config
)
message = """[INST]<<SYS>>You are my assistant<</SYS>>"""
user_input = input("User:")
while user_input != "exit":
    message += user_input + """[/INST]\n"""
    print(message)
    tokens = llm.tokenize(user_input)
    print(tokens)
    start_time = time.time()
    response = llm(
        message,
        stream=False,
    )
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Lama:" + response)
    message += response + "\n"
    user_input = input("User:")
    message += """[INST]"""
