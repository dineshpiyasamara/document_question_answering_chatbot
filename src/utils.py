'''
===========================================
        Module: Util functions
===========================================
'''
import box
import yaml

from langchain import PromptTemplate
from langchain.chains import RetrievalQA, ConversationChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.prompts import qa_template
from src.llm import build_llm

from langchain.memory import ConversationBufferMemory

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question', 'history'])
    return prompt


def build_retrieval_qa(llm, prompt, vectordb, history_list):
    # new_memory = ConversationBufferMemory(memory_key="history",input_key="input")
    # conversation = ConversationChain(
    #     llm=llm,
    #     memory=new_memory,
    #     verbose=False
    # )
    # output = conversation.predict(input='Hello, I love you')
    # print(output)

    memory = ConversationBufferMemory(memory_key="history",input_key="question")
    for history in history_list:
        memory.save_context({'question': history['question']}, 
                            {'output': history['answer']})

    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
                                       return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                       verbose=True,
                                       chain_type_kwargs={
                                           'verbose': True,
                                           'prompt': prompt,
                                           'memory': memory
                                           }
                                       )

    return dbqa


def setup_dbqa(chat_id, history_list):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    vectordb = FAISS.load_local(f'{cfg.DB_FAISS_PATH}/{chat_id}', embeddings)

    llm = build_llm()

    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb, history_list)

    # print("##############################")
    # print(dbqa.combine_documents_chain.memory)

    return dbqa
