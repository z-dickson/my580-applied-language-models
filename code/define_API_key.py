import os

with open('/home/zach/Dropbox/main/dickson_main/API_keys/paid_openAI_key.txt', 'r') as f:
    api_key = f.read().strip()

os.environ['OPENAI_API_KEY'] = api_key
