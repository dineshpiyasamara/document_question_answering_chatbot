'''
===========================================
        Module: Util functions
===========================================
'''
import box
import yaml
import shutil
import os

from langchain import PromptTemplate
from langchain.chains import RetrievalQA, ConversationChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.prompts import qa_template
from src.llm import build_llm
from src.logger import logging
import requests

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

    logging.info("Initialized Conversational Buffered Memory")

    for history in history_list:
        memory.save_context({'question': history['question']}, 
                            {'output': history['answer']})
        
    logging.info("Set chat memory using history")

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

    logging.info("Retrieve chunks from Vector Database")

    # print("##############################")
    # print(dbqa.combine_documents_chain.memory)

    return dbqa

def remove_folder(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        try:
            shutil.rmtree(directory_path)
            logging.info(f"Directory '{directory_path}' has been removed.")
        except OSError as e:
            logging.info(f"Error while removing directory: {e}")
    else:
        logging.info(f"Directory '{directory_path}' does not exist.")


def remove_folder_structure(chat_id):
    directory_path_to_data = f'{cfg.DATA_PATH}{chat_id}'
    directory_path_to_vectordb = f'{cfg.DB_FAISS_PATH}/{chat_id}'

    remove_folder(directory_path_to_data)
    remove_folder(directory_path_to_vectordb)



def run_bot(chat_id, question, history_list):

    # Setup DBQA
    dbqa = setup_dbqa(chat_id, history_list)
    response = dbqa({'query': question})

    logging.info(f'\nAnswer: \n{response["result"]}')
    logging.info('='*50)

    # # Process source documents
    # source_docs = response['source_documents']
    # for i, doc in enumerate(source_docs):
    #     print(f'\nSource Document {i+1}\n')
    #     print(f'Source Text: {doc.page_content}')
    #     print(f'Document Name: {doc.metadata["source"]}')
    #     print(f'Page Number: {doc.metadata["page"]}\n')
    #     print('='* 60)

    return response


def send_request(response):
    url = 'https://example.com/api/endpoint'

    data = {
        'answer': response
    }

    response = requests.post(url, data=data)

    if response.status_code == 200:
        logging.info('POST request successful')
        logging.info('Response:', response.text)
    else:
        logging.info('POST request failed')
        logging.info('Status Code:', response.status_code)