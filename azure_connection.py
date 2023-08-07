from azure.storage.blob import BlobServiceClient
import os
from urllib.parse import urlparse
import box
import yaml

with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

blob_service_client = BlobServiceClient(account_url=f"https://{cfg.AZURE_ACCOUNT_NAME}.blob.core.windows.net", credential=cfg.AZURE_ACCOUNT_KEY)

def download_files(chat_id, file_url_list):
    if not os.path.exists(f'{cfg.DATA_PATH}{chat_id}'):
        os.makedirs(f'{cfg.DATA_PATH}{chat_id}')
        print(f"Folder '{f'{cfg.DATA_PATH}{chat_id}'}' created.")
        for file_url in file_url_list:

            parsed_url = urlparse(file_url)
            path_segments = parsed_url.path.strip("/").split("/")
            container_name = path_segments[0]
            blob_name = "/".join(path_segments[1:])
            file_name = blob_name.split("/")[-1]

            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            with open(f"{cfg.DATA_PATH}{chat_id}/{file_name}", "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
    else:
        print(f"Folder '{f'{cfg.DATA_PATH}{chat_id}'}' already exists.")




