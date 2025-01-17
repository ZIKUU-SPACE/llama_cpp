import streamlit as st 
from langchain.callbacks.base import BaseCallbackManager
from langchain_community.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人アシスタントです。質問に対して日本語で丁寧に回答してください。"

model_path = "./models/gemma-2-2b-it-Q6_K.gguf"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method="markdown"):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

def make_prompt(message):
    prompt = "<start_of_turn>user {system} {prompt} <end_of_turn> <start_of_turn>model".format(system=DEFAULT_SYSTEM_PROMPT, prompt=message)


    return prompt

with st.sidebar:
    st.header('日本語生成AI')
    st.subheader('作る・学ぶ モノづくり塾『ZIKUU』')
    st.divider()

    st.text("設定")
    max_tokens = st.slider('Max Tokens', value=40960, min_value=8192, max_value=81920, step=32)
    temperature = st.slider('Temperature', value=0.8, min_value=0.1, max_value=1.5, step=0.05)
    n_ctx = st.slider("Number of Ctx", value=8192, min_value=64, max_value=8192, step=1)
    n_batch = st.slider("Number of Batch", value=128, min_value=1, max_value=8192, step=1)

with st.form(key="generation_form"):
    prompt = st.text_area('質問')
    do_generate = st.form_submit_button('送信')
    if do_generate and prompt:
        with st.spinner("生成中..."):
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box, display_method='write')
            chat = LlamaCpp(
                model_path=model_path,
                temperature=temperature,
                max_tokens=max_tokens,
                n_ctx=n_ctx,
                n_batch=n_batch,
                callback_manager=BaseCallbackManager([stream_handler]), 
                verbose=False,
            )
            res = chat.invoke(make_prompt(prompt))

