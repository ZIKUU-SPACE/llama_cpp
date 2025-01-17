from llama_cpp import Llama 

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人アシスタントです。質問に対して日本語で丁寧に回答してください。"

def make_prompt(message):
    prompt = "<bos><start_of_turn>user {system} {prompt} <end_of_turn> <start_of_turn>model".format(system=DEFAULT_SYSTEM_PROMPT, prompt=message)
    return prompt

llm = Llama(
        model_path="./models/gemma-2-2b-it-Q6_K.gguf",
        n_ctx=4096,
        )
output = llm(make_prompt("Name the planets in the solar system?. "),
             max_tokens=4096,
             echo=True
             )
print(output)

