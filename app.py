from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import gradio as gr
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import os
# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 下载模型
os.system('huggingface-cli download --resume-download BAAI/bge-base-zh --local-dir /home/xlab-app-center/model/bge-base-zh')
# 将模型导入
from openxlab.model import download
download(model_repo='OpenLMLab/InternLM-chat-7b', output='/home/xlab-app-center/model/InternLM-chat-7b')


def load_chain():
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/home/xlab-app-center/model/bge-base-zh")

    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )

    # 加载自定义 LLM
    llm = InternLM_LLM(model_path = "/home/xlab-app-center/model/InternLM-chat-7b")

    # 定义一个 Prompt Template
    template = """作为一名哲学问答助手，你的任务是根据所提供的上下文回答用户问题。要求如下：
    1.如果上下文与用户问题无关，请回答“对不起，我不知道。”
    2.回答请用中文，保持简洁专业。
    上下文: 
    ···
    {context}
    ···
    用户问题: {question}
    你的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)

    # 运行 chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT}
        )
    
    return qa_chain

class Model_center():
    """
    存储检索问答链的对象 
    """
    def __init__(self):
        # 构造函数，加载检索问答链
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history


if __name__ == "__main__":

    model_center = Model_center()

    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):   
            with gr.Column(scale=15):
                gr.Markdown("""<h1><center>哲学问答小助手</center></h1>
                            <center>基于InternLM-7b</center>
                            """)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=450, show_copy_button=True)
        
                examples = [
                    "马克思主义哲学有两个最显著的特点是什么？",
                    "哲学的基本问题是什么？",
                    "在我国社会主义制度下，什么是最大的生产力？",
                    "列宁说承认什么的才是马克思主义者？",
                    "什么是修正主义？",
                    "如何正确认识爱国主义？",
                    "新黑格尔主义是什么？",
                    "性善论的目的是什么？",
                    "如何评价儒家，如何彻底战胜孔孟之道？",
                    "毛主席说在中国封建社会里，只有什么才是历史发展的真正动力？",
                ]
                msg = gr.Dropdown(choices=examples, label="问题（可选示例）", allow_custom_value=True)

                with gr.Row():
                    # 创建提交按钮。
                    db_wo_his_btn = gr.Button("对话")
                    clear = gr.ClearButton(components=[chatbot], value="清除")

            # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[msg, chatbot], outputs=[msg, chatbot])
            
        gr.Markdown("""提醒：<br>
        1. 受限于模型性能，效果可能会差些。
        2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
        """)
    gr.close_all()
    # 直接启动
    demo.launch()
