import os

# One way to keep your API key private is to save it as a .txt file somewhere locally, and then call that .txt file by updating the file path. If you don't care to keep it private for the workshop, you can replace import define_API_key at the top of the notebooks and use another method highlighted. 

with open('/home/zach/Dropbox/main/dickson_main/API_keys/paid_openAI_key.txt', 'r') as f:
    api_key = f.read().strip()

os.environ['OPENAI_API_KEY'] = api_key
