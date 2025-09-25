
EMBEDDING_URL="http://test.2brain.cn:9800/v1/emb"
RERANK_URL="http://test.2brain.cn:2260/rerank"
IMAGE_MODEL_URL='http://test.2brain.cn:23333/v1'

def local_embedding(inputs):
    """Get embeddings from the embedding service"""
    
    headers = {"Content-Type": "application/json"}
    data = {"texts": inputs}
    
    response = requests.post(EMBEDDING_URL, headers=headers, json=data)
    
    result = response.json()
    return result['data']['text_vectors']


def rerank(query, result_doc):

    res = requests.post(RERANK_URL, json={"query": query, "documents": [doc['text'] for doc in result_doc]}).json()
    if res and 'scores' in res and len(res['scores']) == len(result_doc):
        for idx, doc in enumerate(result_doc):
            result_doc[idx]['score'] = res['scores'][idx]
        
        # Sort documents by rerank score in descending order (highest scores first)
        result_doc.sort(key=lambda x: x['score'], reverse=True)
            
    return result_doc


def summarize_image(image_path, base_url = IMAGE_MODEL_URL):
    retry=0
    while retry<=5:
        try:
            text=f"""
详细地描述这张图片的内容，不要漏掉细节，并提取图片中的文字。注意只需客观说明图片内容，无需进行任何评价。
"""
            # print("prompt:\n",flush=True)
            # print(text+'\n',flush=True)
            # print(image_link)
            client = OpenAI(api_key='YOUR_API_KEY', base_url=base_url)

            # Read local image and convert to Base64 data URL
            with open(image_path, 'rb') as f:
                content_bytes = f.read()
            mime_type = mimetypes.guess_type(image_path)[0] or 'image/png'
            encoded = base64.b64encode(content_bytes).decode('utf-8')
            data_url = f"data:{mime_type};base64,{encoded}"
            resp = client.chat.completions.create(
                model='internvl-internlm2',
                messages=[{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text', 'text': text}, 
                        {
                            'type': 'image_url','image_url': { 'url': data_url}}]
                    }], temperature=0.8, top_p=0.8, max_tokens=2048, stream=False)
            # print(resp.choices[0].message.content)
            
            return resp.choices[0].message.content
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(e)
            time.sleep(1)
            retry+=1
            
    return None


