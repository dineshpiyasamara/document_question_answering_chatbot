import box
import timeit
import yaml
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa
from flask import Flask, request, jsonify
from accelerator import *
from azure_connection import *
from db_build import *

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))



def run_bot(chat_id, question, history_list):

    # Setup DBQA
    start = timeit.default_timer()
    dbqa = setup_dbqa(chat_id, history_list)
    response = dbqa({'query': question})

    end = timeit.default_timer()

    print(f'\nAnswer: \n{response["result"]}')
    print('='*50)

    # # Process source documents
    # source_docs = response['source_documents']
    # for i, doc in enumerate(source_docs):
    #     print(f'\nSource Document {i+1}\n')
    #     print(f'Source Text: {doc.page_content}')
    #     print(f'Document Name: {doc.metadata["source"]}')
    #     print(f'Page Number: {doc.metadata["page"]}\n')
    #     print('='* 60)

    print(f"Time to retrieve response: {end - start}")

    return response

@app.route('/api/test', methods=['GET'])
def test():
    return 'Welcome to Document Answeing bot powered by LLAMA-2'

@app.route('/api/get_answer', methods=['POST'])
def get_answer():
    try:
        data = request.get_json()

        # gpu_conn()

        chat_id = data['chat_id']
        history_list = data['history_list']
        question = data['question']
        file_url_list = data['file_url_list']

        download_files(chat_id, file_url_list)
        run_db_build(chat_id)

        response = run_bot(chat_id, question, history_list)

        data['answer'] = response["result"]
        
        return jsonify({'received_data': data})
    except Exception as e:
        return jsonify({'error': 'Invalid JSON format'}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)