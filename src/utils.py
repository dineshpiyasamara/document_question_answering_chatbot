'''
===========================================
        Module: Util functions
===========================================
'''
import box
import yaml

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
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

memory = ConversationBufferMemory(memory_key="history",input_key="question")

def build_retrieval_qa(llm, prompt, vectordb):

    # memory = ConversationBufferMemory()
    # memory.save_context({'input': 'how handle overfitting?'}, 
    #                     {'output': 'we can use regularization such as l1 and l2 regularization'})

    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
                                       return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
                                    #    verbose=True,
                                       chain_type_kwargs={
                                        #    'verbose': True,
                                           'prompt': prompt,
                                           'memory': memory
                                           }
                                       )
    return dbqa


def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    llm = build_llm()

    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    # print("##############################")
    # print(dbqa.combine_documents_chain.memory)

    return dbqa
