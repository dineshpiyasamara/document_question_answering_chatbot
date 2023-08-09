import box
import yaml
from dotenv import find_dotenv, load_dotenv
from src.utils import *
from src.logger import logging
from flask import Flask, request, jsonify
# from accelerator import *
from azure_connection import *
from db_build import *
import timeit

app = Flask(__name__)

load_dotenv(find_dotenv())

with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def build_folder_structure(chat_id, file_url_list):
    download_files(chat_id, file_url_list)
    run_db_build(chat_id)


@app.route('/api/test', methods=['GET'])
def test():
    return 'Welcome to Document Answeing bot powered by LLAMA-2'

@app.route('/api/update_document', methods=['POST'])
def update_document():
    try:
        data = request.get_json()

        chat_id = data['chat_id']
        file_url_list = data['file_url_list']

        remove_folder_structure(chat_id)

        build_folder_structure(chat_id, file_url_list)
        
        return jsonify({'received_data': data})
    except Exception as e:
        return jsonify({'error': 'Invalid JSON format'}), 400

@app.route('/api/get_answer', methods=['POST'])
def get_answer():
    try:
        start = timeit.default_timer()
        data = request.get_json()

        chat_id = data['chat_id']
        history_list = data['history_list']
        question = data['question']
        file_url_list = data['file_url_list']

        build_folder_structure(chat_id, file_url_list)

        response = run_bot(chat_id, question, history_list)

        data['answer'] = response["result"]

        # send_request(response['result'])

        end = timeit.default_timer()
        logging.info(f"Time to retrieve response: {end - start}")
       
        return jsonify({'received_data': data})
    except Exception as e:
        return jsonify({'error': 'Invalid JSON format'}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)