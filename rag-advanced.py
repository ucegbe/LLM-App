import os
import time
import json
import boto3
import pandas as pd
import logging
import streamlit as st
from pathlib import Path
import time
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfWriter, PdfReader
import requests
import tiktoken
from tqdm import tqdm
import yaml
from anthropic import Anthropic
import sentencepiece
import aspose.words as aw
import multiprocessing
import uuid
from streamlit_chat import message
import fitz
import io
import json
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

st.set_page_config(layout="wide")
logger = logging.getLogger('sagemaker')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

APP_MD    = json.load(open('application_metadata_complete.json', 'r'))
MODELS_LLM = {d['name']: d['endpoint'] for d in APP_MD['models-llm']}
MODELS_EMB = {d['name']: d['endpoint'] for d in APP_MD['models-emb']}
REGION    = APP_MD['region']
BUCKET    = APP_MD['Kendra']['bucket']
PREFIX    = APP_MD['Kendra']['prefix']
OS_ENDPOINT  =  APP_MD['opensearch']['domain_endpoint']
KENDRA_ID = APP_MD['Kendra']['index']
KENDRA_ROLE=APP_MD['Kendra']['role']
PARENT_TEMPLATE_PATH="prompt_template"
KENDRA_S3_DATA_SOURCE_NAME=APP_MD['Kendra']['s3_data_source_name']
try:
    DYNAMODB_TABLE=APP_MD['dynamodb_table']
    DYNAMODB_USER=APP_MD['dynamodb_user']
except:
    DYNAMODB_TABLE=""
    

DYNAMODB      = boto3.resource('dynamodb')
S3            = boto3.client('s3', region_name=REGION)
TEXTRACT      = boto3.client('textract', region_name=REGION)
KENDRA        = boto3.client('kendra', region_name=REGION)
SAGEMAKER     = boto3.client('sagemaker-runtime', region_name=REGION)
BEDROCK = boto3.client(service_name='bedrock-runtime',region_name='us-east-1') 
COMPREHEND=boto3.client("comprehend")

# Vector dimension mappings of each embedding model
EMB_MODEL_DICT={"titan":1536,
                "minilmv2":384,
                "bgelarge":1024,
                "gtelarge":1024,
                "e5largev2":1024,
                "e5largemultilingual":1024,
               "gptj6b":4096,
                "cohere":1024}

# Creating unique domain names for each embedding model using the domain name prefix set in the config json file
# and a corresponding suffix of the embedding model name
EMB_MODEL_DOMAIN_NAME={"titan":f"{APP_MD['opensearch']['domain_name']}_titan",
                "minilmv2":f"{APP_MD['opensearch']['domain_name']}_minilm",
                "bgelarge":f"{APP_MD['opensearch']['domain_name']}_bgelarge",
                "gtelarge":f"{APP_MD['opensearch']['domain_name']}_gtelarge",
                "e5largev2":f"{APP_MD['opensearch']['domain_name']}_e5large",
                "e5largemultilingual":f"{APP_MD['opensearch']['domain_name']}_e5largeml",
               "gptj6b":f"{APP_MD['opensearch']['domain_name']}_gptj6b",
                       "cohere":f"{APP_MD['opensearch']['domain_name']}_cohere"}

# Session state keys
if 'generate' not in st.session_state:
    st.session_state['generate'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'domain' not in st.session_state:
    st.session_state['domain'] = ""
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = ''
if 'token' not in st.session_state:
    st.session_state['token'] = 0
if 'text' not in st.session_state:
    st.session_state['text'] = ''
if 'summary' not in st.session_state:
    st.session_state['summary']=''
if 'message' not in st.session_state:
    st.session_state['message'] = []
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 0   
if 'bytes' not in st.session_state:
    st.session_state['bytes'] = None
if 'rtv' not in st.session_state:
    st.session_state['rtv'] = ''
if 'page_summ' not in st.session_state:
    st.session_state['page_summ'] = ''
if 'action_name' not in st.session_state:
    st.session_state['action_name'] = ""
if 'chat_memory' not in st.session_state:
    if DYNAMODB_TABLE:
        chat_histories = DYNAMODB.Table(DYNAMODB_TABLE).get_item(Key={"UserId": DYNAMODB_USER})
        if "Item" in chat_histories:
            st.session_state['chat_memory']=chat_histories['Item']['messages']
        else:
            st.session_state['chat_memory']=[]
    else:
        st.session_state['chat_memory'] = []



def query_index(query):  
    response = KENDRA.retrieve(
    IndexId=KENDRA_ID,
    QueryText=query,
    )
    return response

@st.cache_resource
def token_counter(path):
    tokenizer = AutoTokenizer.from_pretrained(path, token=APP_MD['hugginfacekey'])
    return tokenizer

@st.cache_resource
def token_cohere(path):
    tokenizer = Tokenizer.from_pretrained(path)
    return tokenizer

def create_os_index(param, chunks):
    """ Create an Opensearch Index
        It uses four mappings:
        - embedding: chunk vector embedding
        - passage_id: document page number of chunk
        - passage: chunk
        - doc_id: name of document
        
        An opensearch undex is created per embedding model selected every subsequest indexing using that model, goes to the same opensearch index.
        To use a new index, change teh opensearch domain name in the configuration json file.
    """
    st.write("Indexing...")    
    domain_endpoint = OS_ENDPOINT
    service = 'es'
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, REGION, service, session_token=credentials.token)
    os_ = OpenSearch(
        hosts = [{'host': OS_ENDPOINT, 'port': 443}],
        http_auth = awsauth,
        use_ssl = True,
        verify_certs = True,
        timeout=120,        
        # http_compress = True, # enables gzip compression for request bodies
        connection_class = RequestsHttpConnection
    )

    mapping = {
      'settings': {
        'index': {  
          'knn': True,
          "knn.algo_param.ef_search": round(float(param["ef_search"])),
             "knn.space_type": "cosinesimil",
        }
          },

          'mappings': {  
            'properties': {
              'embedding': {
                'type': 'knn_vector', 
                'dimension': EMB_MODEL_DICT[param['emb'].lower()],
                "method": {
                  "name": "hnsw",       
                  "space_type": "l2",
                  "engine": param["engine"],
                  "parameters": {
                     "ef_construction": round(float(param["ef_construction"])),
                     "m":  round(float(param["m"]))
                   }
                }
              },

              'passage_id': {
                'type': 'keyword'
              },

              'passage': {
                'type': 'text'
              },

              'doc_id': {
                'type': 'keyword'
              }
            }
          }
        }

    domain_index = f"{param['domain']}_{param['engine']}"    
    
    if not os_.indices.exists(index=domain_index):        
        os_.indices.create(index=domain_index, body=mapping)
        # Verify that the index has been created
        if os_.indices.exists(index=domain_index):
            st.write(f"Index {domain_index} created successfully.")
        else:
            st.write(f"Failed to create index '{domain_index}'.")
    else:
        st.write(f'{domain_index} Index already exists!')
        
    i = 1
  
    for pages, chunk in chunks.items(): # Iterate through dict with chunk page# and content
        chunk_id = pages.split('*')[0] # take care of multiple chunks in same page (*) is used as delimiter
        if "titan" in param["emb"].lower():
            prompt= {
                "inputText": chunk
                }

            body=json.dumps(prompt)
            modelId = param["emb_model"]
            accept = 'application/json'
            contentType = 'application/json'

            response = BEDROCK.invoke_model(body=body, modelId=modelId, accept=accept,contentType=contentType)
            response_body = json.loads(response.get('body').read())
            embedding=response_body['embedding']
        elif "cohere" in param["emb"].lower(): 
            prompt= {
                "texts": [chunk],
             "input_type": "search_document"
            }
            body=json.dumps(prompt)
            modelId = param["emb_model"]
            accept = "*/*"
            contentType = 'application/json'

            response = BEDROCK.invoke_model(body=body, modelId=modelId, accept=accept,contentType=contentType)
            response_body = json.loads(response.get('body').read())
            embedding=response_body['embeddings'][0]
        else:            
            payload = {'text_inputs': [chunk]}
            payload = json.dumps(payload).encode('utf-8')

            response = SAGEMAKER.invoke_endpoint(EndpointName=param["emb_model"], 
                                                        ContentType='application/json',  
                                                        Body=payload)

            model_predictions = json.loads(response['Body'].read())
            embedding = model_predictions['embedding'][0]
        document = { 
            'doc_id': st.session_state['file_name'], 
            'passage_id': chunk_id,
            'passage': chunk, 
            'embedding': embedding}
        try:
            response = os_.index(index=domain_index, body=document)
            i += 1
            # Check the response to see if the indexing was successful
            if response["result"] == "created":
                print(f"Document indexed successfully with ID: {response['_id']}")
            else:
                print("Failed to index document.")
        except RequestError as e:
            logging.error(f"Error indexing document to index '{domain_index}': {e}")
    return domain_index


def kendra_index(doc_name):
    """Create kendra s3 data source and sync files into kendra index"""
    import time
    response=KENDRA.list_data_sources(IndexId=KENDRA_ID)['SummaryItems']
    data_sources=[x["Name"] for x in response if KENDRA_S3_DATA_SOURCE_NAME in x["Name"]]
    if data_sources: # Check if s3 data source already exist and sync files
        data_source_id=[x["Id"] for x in response if KENDRA_S3_DATA_SOURCE_NAME in x["Name"] ][0]
        sync_response = KENDRA.start_data_source_sync_job(
        Id = data_source_id,
        IndexId =KENDRA_ID
        )    
        status=True
        while status:
            jobs = KENDRA.list_data_source_sync_jobs(
                Id = data_source_id,
                IndexId = KENDRA_ID
            )
            # For this example, there should be one job        
            try:
                status = jobs["History"][0]["Status"]
                st.write(" Syncing data source. Status: "+status)
                if status != "SYNCING":
                    status=False
                time.sleep(2)
            except:
                time.sleep(2)
    else: # Create a Kendra s3 data source and sync files
        
        index_id=KENDRA_ID
        response = KENDRA.create_data_source(
            Name=KENDRA_S3_DATA_SOURCE_NAME,
            IndexId=index_id,
            Type='S3',
            Configuration={
                'S3Configuration': {
                    'BucketName': BUCKET,
                    'InclusionPrefixes': [
                        f"{PREFIX}/",
                    ],            
                },
            },     
            RoleArn=KENDRA_ROLE, 
            ClientToken=doc_name,                
        )    
        data_source_id=response['Id']
        import time
        status=True
        while status:
            # Get the details of the data source, such as the status
            data_source_description = KENDRA.describe_data_source(
                Id = data_source_id,
                IndexId = index_id
            )
            # If status is not CREATING, then quit
            status = data_source_description["Status"]
            st.write(" Creating data source. Status: "+status)
            time.sleep(2)
            if status != "CREATING":
                status=False            
        sync_response = KENDRA.start_data_source_sync_job(
            Id = data_source_id,
            IndexId = index_id
        )    
        status=True
        while status:
            jobs = KENDRA.list_data_source_sync_jobs(
                Id = data_source_id,
                IndexId = index_id
            )
                   
            try:
                status = jobs["History"][0]["Status"]
                st.write(" Syncing data source. Status: "+status)
                if status != "SYNCING":
                    status=False
                time.sleep(2)
            except:
                time.sleep(2)
            
def get_chunk_pages(page_dict,chunk):
    """
    Getting chunk page number of each chunk to use as metadata while creating the opensearch index.
    """
    token_dict={}
    length=0
    for pages in page_dict.keys():         
        length+=len(page_dict[pages].split())       
        token_dict[pages]=length
    chunk_page={}
    cumm_chunk=chunk
    for page, token_size in token_dict.items():
        while True:        
            if token_size%cumm_chunk==token_size:
                try:
                    chunk_page[round(cumm_chunk/chunk)]+=f'_{page}'
                except:
                    chunk_page[round(cumm_chunk/chunk)]=str(page)
                break
            elif token_size%cumm_chunk==0:
                try:
                    chunk_page[round(cumm_chunk/chunk)]+=f'_{page}'
                except:
                    chunk_page[round(cumm_chunk/chunk)]=str(page)
                cumm_chunk=+chunk
                break
            else:
                try:
                    chunk_page[round(cumm_chunk/chunk)]+=f'_{page}'
                except:
                    chunk_page[round(cumm_chunk/chunk)]=str(page)
                cumm_chunk+=chunk
    return chunk_page

def chunker(chunk_size, file):
    """
    Chunking by number of words, rule of thumb (1 token is ~3/5th a word).
    I did not clean punctuation marks or do any text cleaning.
    """
    chunk_size=round(chunk_size)
    result={}      
    text=' '.join(file.values())    
    words=text.split()
    n_docs = 1    
    chunk_pages=get_chunk_pages(file,chunk_size) # get page number for each chunk to use as metadata
    for i in range(0, len(words), chunk_size): # iterate through doc and create chunks not exceeding chunk size       
        chunk_words = words[i: i+chunk_size]   
        chunk = ' '.join(chunk_words)
        if chunk_pages[(int(i)//chunk_size)+1] in result.keys():
            result[f"{chunk_pages[(int(i)//chunk_size)+1]}*{str(time.time()).split('.')[-1]}"]=chunk
        else:
            result[chunk_pages[(int(i)//chunk_size)+1]]=chunk
        n_docs += 1    
    return result

def full_doc_extraction(file):    
    """ This is the function for the full-page retrieval technique.
        It uses the metadata collected from the retrieval system to get the original source document 
        and extract the full page content, if pdf, or entire document content (txt, json, xml files etc.)
        
        You can append functions to handle any other file format including web bages.
    """
    link=file.split('###')[0]
    if "pdf" in os.path.splitext(link)[-1]:
        files=file.split('###')
        bucket=files[0].split('/',4)[3]
        key=files[0].split('/',4)[-1]
        
        ## Read pdf file in memory
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, key)
        pdf_bytes=obj.get()['Body']
        
        ## open pdf file 
        with io.BytesIO(pdf_bytes.read()) as open_pdf_file:   
            doc = fitz.open(stream=open_pdf_file)  
        if doc.page_count>1:    
            ## Extract the page from pdf file and send to textract
            page = doc.load_page(int(files[-1]))  
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
            response = TEXTRACT.detect_document_text(
            Document={      
                 'Bytes': pix.tobytes("png", 100),
                }
            )
            blocks = response['Blocks']   

            bbox=[]
            word=[]
            for item in response['Blocks']:
                if item["BlockType"] == "WORD"  :
                    # bbox.append(item['Geometry']['BoundingBox'])
                    word.append(item["Text"])    

            result=" ".join([x for x in word])
        else:
            ## pdf has one page, send to textract
            response = TEXTRACT.detect_document_text(
            Document={      
                'S3Object': {
                'Bucket': bucket,
                'Name': key,         
                }
                }
            )
            blocks = response['Blocks']   

            bbox=[]
            word=[]
            for item in response['Blocks']:
                if item["BlockType"] == "WORD"  :
                    # bbox.append(item['Geometry']['BoundingBox'])
                    word.append(item["Text"])    

            result=" ".join([x for x in word])           
    
    elif "txt" in os.path.splitext(link)[-1]:
        files=file.split('###')
        bucket=files[0].split('/',4)[3]
        key=files[0].split('/',4)[-1]
        
        ## Read pdf file in memory
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, key)
        result=obj.get()['Body'].read().decode()
        
        
    elif "html" in os.path.splitext(link)[-1]:
        files=file.split('###')
        bucket=files[0].split('/',4)[3]
        key=files[0].split('/',4)[-1]

        ## Read pdf file in memory
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket, key)
        html=obj.get()['Body'].read().decode()  
        
        part1=html.split('<head>')
        part2=part1[-1].split('</head>')
        result=part1[0]+part2[1]
      
    elif "xml" in os.path.splitext(link)[-1]:
        pass
    elif "csv" in os.path.splitext(link)[-1]:
        pass
    
    return result

def retrieval_quality_check(doc, question):
    template=f"""\n\nHuman:
Here is a document:
<document>
{doc}
</document>      

Here is a question:
Question: {question}

Review the document and check if the document is sufficient enough to answer the question completely.
If the complete answer is contained in the document respond with:
<answer>
yes
</answer>

Else respond with:
<answer>
no
</answer>

Your response should not include any preamble, just provide the answer alone in your response.\n\nAssistant:"""

    prompt={
      "prompt": template,
      "max_tokens_to_sample": 10,
      "temperature": 0.1,
      # "top_k": 250,
      # "top_p": 1,  
      #    "stop_sequences": []
    }
    prompt=json.dumps(prompt)
    output = BEDROCK.invoke_model(body=prompt,
                                    modelId='anthropic.claude-v2',  #Change model ID to a diffent anthropic model id
                                    accept="application/json", 
                                    contentType="application/json")

    output=output['body'].read().decode()
    answer=json.loads(output)['completion']
    idx1 = answer.index('<answer>')
    idx2 = answer.index('</answer>')
    response=answer[idx1 + len('<answer>') + 1: idx2]
    print(response)
    return response

def single_passage_retrieval(responses,prompt,params, handler=None):
    """
    Sends one retrieved passage a time per TopK selected to the LLM together with the user prompt.
    """
    models=["claude","llama","cohere","ai21","titan","mistral"] # mapping for the prompt template stored locally
    chosen_model=[x for x in models if x in params['model_name'].lower()][0]
    total_response=[]   
    # Assigning roles dynamically to the LLM based on persona
    persona_dict={"General":"assistant","Finance":"finanacial analyst","Insurance":"insurance analyst","Medical":"medical expert"}
    with open(f"{PARENT_TEMPLATE_PATH}/rag/{chosen_model}/prompt1.txt","r") as f:
            prompt_template=f.read()
    if "Kendra" in params["rag"]:
        for x in range(0, round(params['K'])): # provide an answer for each number of passage retrieved by the Retriever
            score = responses['ResultItems'][x]['ScoreAttributes']["ScoreConfidence"]              
            passage = responses['ResultItems'][x]['Content']
            doc_link = responses['ResultItems'][x]['DocumentURI']
            page_no=""                
            if os.path.splitext(doc_link)[-1]:                    
                page_no=responses['ResultItems'][x]['DocumentAttributes'][1]['Value']['LongValue']
            s3_uri=responses['ResultItems'][x]['DocumentId']
            doc_name=doc_link.split('/')[-1] if doc_link.split('/')[-1] else doc_link.split('/')[-2]
            qa_prompt =prompt_template.format(doc=passage, prompt=prompt, role=persona_dict[params['persona']])
            if "claude" in params['model_name'].lower():
                qa_prompt=f"\n\nHuman:\n{qa_prompt}\n\nAssistant:"  

            answer, in_token, out_token=query_endpoint(params, qa_prompt,handler)         
            answer1={'Answer':answer, 'Name':doc_name,'Link':doc_link, 'Score':score, 'Page': page_no, "Input Token":in_token,"Output Token":out_token}
            total_response.append(answer1)
    elif "OpenSearch" in params["rag"]:
        for response in responses:            
            score = response['_score']
            passage = response['_source']['passage']
            doc_link = f"https://{BUCKET}.s3.amazonaws.com/file_store/{response['_source']['doc_id']}"
            page_no = response['_source']['passage_id']   
            doc_name = response['_source']['doc_id']   
            qa_prompt =prompt_template.format(doc=passage, prompt=prompt, role=persona_dict[params['persona']])
            if "claude" in params['model_name'].lower():
                qa_prompt=f"\n\nHuman:\n{qa_prompt}\n\nAssistant:"  
            answer, in_token, out_token=query_endpoint(params, qa_prompt,handler)
            answer1={'Answer':answer, "Name":doc_name,'Link':doc_link, 'Score':score, 'Page': page_no,"Input Token":in_token,"Output Token":out_token}
            total_response.append(answer1)

    if params["memory"]:
        chat_history={"user" :prompt,
        "assistant":answer}           
        if DYNAMODB_TABLE:
            put_db(chat_history)
        else:
            st.session_state['chat_memory'].append(chat_history)            
    return total_response

def combined_passages_retrieval_technique(responses,prompt,params, handler=None):
    """
    Function implements the combined passaged retrieval technique.
    It combines topK retrieved passages into a single context and pass to the text LLM together with the user prompt.
    """
    models=["claude","llama","cohere","ai21","titan","mistral"] # mapping for the prompt template stored locally
    chosen_model=[x for x in models if x in params['model_name'].lower()][0]
    # Assigning roles dynamically to the LLM based on persona
    persona_dict={"General":"assistant","Finance":"finanacial analyst","Insurance":"insurance analyst","Medical":"medical expert"}  
    
    if "Kendra" in params["rag"]:
        score = ", ".join([x['ScoreAttributes']["ScoreConfidence"]   for x in responses['ResultItems'][:round(params['K'])]]) 
        doc_link = [x['DocumentURI'] for x in responses['ResultItems'][:round(params['K'])]]   
        page_no=", ".join([str(x['DocumentAttributes'][1]['Value']['LongValue']) for x in responses['ResultItems'][:round(params['K'])] if os.path.splitext(x['DocumentURI'])[-1]])
        doc_name=doc_name = ", ".join([x.split('/')[-1] if x.split('/')[-1] else x.split('/')[-2] for x in doc_link])
        # page_no = ", ".join([x.split('/')[-1].split('-')[-1].split('.')[0] for x in doc_link])              

        holder={}
        for x,y in enumerate(responses['ResultItems'][:round(params['K'])]):
            holder[x]={"document":y["Content"], "source":y['DocumentURI']}  
        # Reading the prompt template based on chosen model and assigning the model roles based on chosen persona
        with open(f"{PARENT_TEMPLATE_PATH}/rag/{chosen_model}/prompt1.txt","r") as f:
            prompt_template=f.read()    
        qa_prompt =prompt_template.format(doc=holder, prompt=prompt, role=persona_dict[params['persona']])
        if "claude" in params['model_name'].lower():
            qa_prompt=f"\n\nHuman:\n{qa_prompt}\n\nAssistant:"              
        answer, in_token, out_token=query_endpoint(params, qa_prompt,handler)
        answer1={'Answer':answer, 'Name':doc_name,'Link':doc_link, 'Score':score, 'Page': page_no, "Input Token":in_token,"Output Token":out_token}
        
    elif "OpenSearch" in params["rag"]:
        with open(f"{PARENT_TEMPLATE_PATH}/rag/{chosen_model}/prompt1.txt","r") as f:
                        prompt_template=f.read()
        score = ",".join([str(x['_score']) for x in responses])
        passage = [x['_source']['passage'] for x in responses]
        doc_link = f"https://{BUCKET}.s3.amazonaws.com/file_store/{responses[0]['_source']['doc_id']}"
        page_no = ",".join([x['_source']['passage_id'] for x in responses])
        doc_name = ", ".join([x['_source']['doc_id'] for x in responses])
        qa_prompt =prompt_template.format(doc=passage, prompt=prompt, role=persona_dict[params['persona']])
        if "claude" in params['model_name'].lower():
            qa_prompt=f"\n\nHuman:\n{qa_prompt}\n\nAssistant:"  
        answer, in_token, out_token=query_endpoint(params, qa_prompt,handler)
        answer1={'Answer':answer, "Name":doc_name,'Link':doc_link, 'Score':score, 'Page': page_no,"Input Token":in_token,"Output Token":out_token}

    # Saving chat history if enabled
    if params["memory"]:
        chat_history={"user" :prompt,
        "assistant":answer}
        # Store chat history in DynamoDb or in-memory
        if DYNAMODB_TABLE:
            put_db(chat_history)
        else:
            st.session_state['chat_memory'].append(chat_history)
    return answer1

def full_page_retrieval_technique_kendra(responses,prompt,params, handler=None):
    """
    Function implements the full-page retrieval technique.
    It gets the metatdata (page number, doc location etc) from the topK retrieved responses.
    extracts the entire page (if pdf) or entire doc (json, txt, etc.), combines all topK extraction into a single context and passes to the LLM
    together with the user prompt.
    """
    models=["claude","llama","cohere","ai21","titan","mistral"] # mapping for the prompt template stored locally
    chosen_model=[x for x in models if x in params['model_name'].lower()][0]

    # Assigning roles dynamically to the LLM based on persona
    persona_dict={"General":"assistant","Finance":"finanacial analyst","Insurance":"insurance analyst","Medical":"medical expert"}    
    score = ", ".join([x['ScoreAttributes']["ScoreConfidence"]   for x in responses['ResultItems'][:round(params['K'])]]) 
    doc_link = [x['DocumentURI'] for x in responses['ResultItems'][:round(params['K'])]]   
    doc_name=", ".join([x.split('/')[-1] for x in doc_link])
    sources = []  

    page_no=[str(x['DocumentAttributes'][1]['Value']['LongValue']) for x in responses['ResultItems'][:round(params['K'])] if os.path.splitext(x['DocumentURI'])[-1]]
    holder={}           

    from itertools import zip_longest                  
    for item1, item2 in zip_longest(doc_link, page_no, fillvalue=''):
        sources.append('###'.join([str(item1), str(item2)]))
    page_no=", ".join(page_no)   

    ## Parallely extracting the full document pages
    import multiprocessing    
    num_concurrent_invocations = len(sources)
    pool = multiprocessing.Pool(processes=num_concurrent_invocations)            
    context=pool.map(full_doc_extraction, sources)
    pool.close()
    pool.join() 

    with open(f"{PARENT_TEMPLATE_PATH}/rag/{chosen_model}/prompt1.txt","r") as f:
        prompt_template=f.read()    
    qa_prompt =prompt_template.format(doc=context, prompt=prompt, role=persona_dict[params['persona']])
    if "claude" in params['model_name'].lower():
        qa_prompt=f"\n\nHuman:\n{qa_prompt}\n\nAssistant:"  

    answer, in_token, out_token=query_endpoint(params, qa_prompt,handler)
    answer1={'Answer':answer, 'Name':doc_name,'Link':doc_link, 'Score':score, 'Page': page_no, "Input Token":in_token,"Output Token":out_token}

    if params["memory"]:
        chat_history={"user" :prompt,
        "assistant":answer}           
        if DYNAMODB_TABLE:
            put_db(chat_history)
        else:
            st.session_state['chat_memory'].append(chat_history)
    return answer1


def doc_qna_endpoint(endpoint, responses,prompt,params, handler=None):
    """This function implements the retrieval technique selected for each retriever, collects retrieved passage metadata and queries the 
        Text generation LLm endpoint.
    """
    total_response=[]  
    
    # If auto retriever technique is selected, the an interim LLM call is made to check the quality of a single retrieved passage
    if params["auto_rtv"]:          
        passage = responses['ResultItems'][0]['Content']
        q_c=retrieval_quality_check(passage, prompt)
        # If the single passage contains sufficient information, pass to the final LLM to get a response
        if "yes" in q_c:
            params["K"]=1
            total_response=single_passage_retrieval(responses,prompt,params, handler)                      
            return total_response
        else:
            # If not, call the full_page_retriever technique with topK=3
            params["K"]=3
            answer1=full_page_retrieval_technique_kendra(responses,prompt,params, handler)
            total_response.append(answer1)
            return total_response
    
    else:
        if "Kendra" in params["rag"]:
            if 'combined-passages' in params['method']:
                answer1=combined_passages_retrieval_technique(responses,prompt,params, handler)
                total_response.append(answer1)
                return total_response
            elif 'full-pages' in params['method']:            
                answer1=full_page_retrieval_technique_kendra(responses,prompt,params, handler)
                total_response.append(answer1)
                return total_response
            elif 'single-passages' in params['method']:
                total_response=single_passage_retrieval(responses,prompt,params, handler)
                return total_response

        elif "OpenSearch" in params["rag"]:
            if 'single-passages'  in params['method']:
                total_response=single_passage_retrieval(responses,prompt,params, handler)
                return total_response
            elif 'combined-passages' or 'full-pages' in params['method']:
                answer1=combined_passages_retrieval_technique(responses,prompt,params, handler)
                total_response.append(answer1)
                return total_response

def llm_memory(question, params=None):
    """ This function determines the context of each new question looking at the conversation history...
        to send the appropiate question to the retriever.    
        Messages are stored in DynamoDb if a table is provided or in memory, in the absence of a provided DynamoDb table
    """
    if DYNAMODB_TABLE:
        chat_histories = DYNAMODB.Table(DYNAMODB_TABLE).get_item(Key={"UserId": DYNAMODB_USER})
        if "Item" in chat_histories:
            st.session_state['chat_memory']=chat_histories['Item']['messages']
        else:
            st.session_state['chat_memory']=[]    
    
    chat_string = ""
    for entry in st.session_state['chat_memory']:
        chat_string += f"user: {entry['user']}\nassistant: {entry['assistant']}\n"
    memory_template = f"""\n\nHuman:
Here is the history of your conversation dialogue with a user:
<history>
{chat_string}
</history>

Here is a new question from the user:
user: {question}

Your task is to determine if the question is a follow-up to the previous conversation:
- If it is, rephrase the question as an independent question while retaining the original intent.
- If it is not, respond with "_no_".

Remember, your role is not to answer the question!

Format your response as:
<response>
answer
</response>\n\nAssistant:"""
    if chat_string:
        inference_modifier = {'max_tokens_to_sample':70, 
                              "temperature":0.1,                   
                             }
        llm = Bedrock(model_id='anthropic.claude-v2', client=BEDROCK, model_kwargs = inference_modifier,
                      streaming=False,  # Toggle this to turn streaming on or off
                      callbacks=[StreamingStdOutCallbackHandler() ])
        answer=llm(memory_template)
        idx1 = answer.index('<response>')
        idx2 = answer.index('</response>')
        question_2=answer[idx1 + len('<response>') + 1: idx2]
        if '_no_' not in question_2:
            question=question_2
        print(question)
    return question

def put_db(messages):
    """Store long term chat history in DynamoDB"""
    
    chat_item = {
        "UserId": DYNAMODB_USER,
        "messages": [messages]  # Assuming 'messages' is a list of dictionaries
    }

    # Check if the user already exists in the table
    existing_item = DYNAMODB.Table(DYNAMODB_TABLE).get_item(Key={"UserId": DYNAMODB_USER})

    # If the user exists, append new messages to the existing ones
    if "Item" in existing_item:
        existing_messages = existing_item["Item"]["messages"]
        chat_item["messages"] = existing_messages + [messages]

    response = DYNAMODB.Table(DYNAMODB_TABLE).put_item(
        Item=chat_item
    )    


def llm_streamer():
    output = ""
    i = 1
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                text = chunk_obj["generation"]
                output+=text
                print(f'{text}', end="")
                i+=1



def query_endpoint(params, qa_prompt, handler=None):
    """
    Function to query the LLM endpoint and count tokens in and out.
    """
    if 'ai21' in params['model_name'].lower():
        import json        
        prompt={
          "prompt":  qa_prompt,
          "maxTokens": params['max_len'],
          "temperature": round(params['temp'],2),
          "topP":  params['top_p'], 
        }
    
        prompt=json.dumps(prompt)
        response = BEDROCK.invoke_model(body=prompt,
                                modelId=params['endpoint-llm'], 
                                accept="application/json", 
                                contentType="application/json")
        answer=response['body'].read().decode() 
        answer=json.loads(answer)
        input_token=len(answer['prompt']['tokens'])
        output_token=len(answer['completions'][0]['data']['tokens'])
        tokens=input_token+output_token
        answer=answer['completions'][0]['data']['text']
        st.session_state['token']+=tokens

    elif 'claude' in params['model_name'].lower():
        import json       
        qa_prompt=f"\n\nHuman:\n{qa_prompt}\n\nAssistant:"
        prompt={
          "prompt": qa_prompt,
          "max_tokens_to_sample": round(params['max_len']),
          "temperature": params['temp'],
          # "top_k": 250,
          "top_p":params['top_p'],  
             # "stop_sequences": []
        }
        prompt=json.dumps(prompt)
        response = BEDROCK.invoke_model_with_response_stream(body=prompt, modelId=params['endpoint-llm'], accept="application/json",  contentType="application/json")
        stream = response.get('body')
        answer = ""
        i = 1
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    text = chunk_obj['completion']
                    answer+=text
                    handler.markdown(answer.replace("$","USD ").replace("%", " percent"))                  
                    i+=1      
        claude = Anthropic()
        input_token=claude.count_tokens(qa_prompt)
        output_token=claude.count_tokens(answer)
        tokens=claude.count_tokens(f"{qa_prompt} {answer}")
        st.session_state['token']+=tokens

    elif 'titan' in params['model_name'].lower():      
        import json
        encoding = tiktoken.get_encoding('cl100k_base') #using openai tokenizer, replace with Titan's tokenizer        
        prompt={
               "inputText": qa_prompt,
               "textGenerationConfig": {
                   "maxTokenCount": params['max_len'],     
                   "temperature":params['temp'],
                   "topP":params['top_p'],  
                   },
            }
        prompt=json.dumps(prompt)
        response = BEDROCK.invoke_model_with_response_stream(body=prompt, modelId=params['endpoint-llm'], accept="application/json",  contentType="application/json")
        stream = response.get('body')
        answer = ""
        i = 1
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    text = chunk_obj["outputText"]
                    answer+=text
                    handler.markdown(answer.replace("$","USD ").replace("%", " percent"))                  
                    i+=1   
        input_token=len(encoding.encode(qa_prompt))
        output_token=len(encoding.encode(answer))
        tokens=len(encoding.encode(f"{qa_prompt} {answer}"))
        st.session_state['token']+=tokens
        
    elif 'cohere' in params['model_name'].lower():
        import json         
        prompt={
          "prompt": qa_prompt,
          "max_tokens": round(params['max_len']), 
          "temperature": params['temp'], 
          "return_likelihoods": "GENERATION"   
        }
        prompt=json.dumps(prompt)
        response = BEDROCK.invoke_model_with_response_stream(body=prompt, modelId=params['endpoint-llm'], accept="application/json",  contentType="application/json")
        stream = response.get('body')
        answer = ""
        i = 1
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    text = chunk_obj["generations"][0]['text']
                    answer+=text
                    handler.markdown(answer.replace("$","USD ").replace("%", " percent"))                  
                    i+=1    
        encoding=token_cohere("Cohere/command-nightly")
        input_token=len(encoding.encode(qa_prompt))
        output_token=len(encoding.encode(answer))
        tokens=len(encoding.encode(f"{qa_prompt} {answer}"))
        st.session_state['token']+=tokens   


    elif 'mistral' in params['model_name'].lower(): 
        import json
        payload = {
               "inputs": qa_prompt,
                "parameters": {"max_new_tokens": params['max_len'], 
                               "top_p": params['top_p'] if params['top_p']<1 else 0.99 ,
                               "temperature": params['temp'] if params['temp']>0 else 0.01,
                               "return_full_text": False,}
            } 
        output=SAGEMAKER.invoke_endpoint(Body=json.dumps(payload), EndpointName=params['endpoint-llm'],ContentType="application/json")
        answer=json.loads(output['Body'].read().decode())[0]['generated_text']
        tkn=token_counter("mistralai/Mistral-7B-v0.1")
        input_token=len(tkn.encode(qa_prompt))
        output_token=len(tkn.encode(answer))       
        tokens=len(tkn.encode(f"{qa_prompt} {answer}"))
        st.session_state['token']+=tokens    

    elif 'llama2' in params['model_name'].lower():   
        import json 
        if "bedrock" in params['model_name'].lower(): 
            
            prompt={
              "prompt": qa_prompt,
                    "max_gen_len":params['max_len'], 
                    "temperature":params['temp'] if params['temp']>0 else 0.01,
                    "top_p": params['top_p'] if params['top_p']<1 else 0.99 
            }
            prompt=json.dumps(prompt)
            # st.write(prompt)
            response = BEDROCK.invoke_model_with_response_stream(body=prompt, modelId=params['endpoint-llm'], accept="application/json",  contentType="application/json")
            stream = response.get('body')
            answer = ""
            i = 1
            if stream:
                for event in stream:
                    chunk = event.get('chunk')
                    if chunk:
                        chunk_obj = json.loads(chunk.get('bytes').decode())
                        text = chunk_obj["generation"]
                        answer+=text
                        handler.markdown(answer.replace("$","USD ").replace("%", " percent"))                  
                        i+=1      
        else:            
            payload = {
               "inputs": qa_prompt,
                "parameters": {"max_new_tokens": params['max_len'], 
                               "top_p": params['top_p'] if params['top_p']<1 else 0.99 ,
                               "temperature": params['temp'] if params['temp']>0 else 0.01,
                               "return_full_text": False,}
            }
            output=SAGEMAKER.invoke_endpoint(Body=json.dumps(payload), EndpointName=params['endpoint-llm'],ContentType="application/json",CustomAttributes='accept_eula=true')
            answer=json.loads(output['Body'].read().decode())[0]['generated_text']
        tkn=token_counter('meta-llama/Llama-2-13b-chat-hf')
        input_token=len(tkn.encode(qa_prompt))
        output_token=len(tkn.encode(answer))       
        tokens=len(tkn.encode(f"{qa_prompt} {answer}"))
        st.session_state['token']+=tokens    
    return answer, input_token, output_token
        

def load_document(file_bytes, doc_name, param):
     
    if "Kendra" in param["rag"]:    
       
        dir_name=os.path.splitext(doc_name)[0]
        S3.upload_fileobj(file_bytes, BUCKET, f"{PREFIX}/{dir_name}/{doc_name}")
        time.sleep(1)
        st.write('Kendra Indexing')
        kendra_index(dir_name) 
       
    elif "OpenSearch" in param["rag"]:
        s3_path=f"file_store/{doc_name}"
        file_bytes=file_bytes.read()
        S3.put_object(Body=file_bytes,Bucket= BUCKET, Key=s3_path)       
        time.sleep(1)    
        doc_name=os.path.splitext(doc_name)[0]
        with io.BytesIO(file_bytes) as open_pdf_file:   
            doc = fitz.open(stream=open_pdf_file) 
        if doc.page_count>1:    
            text= extract_text(BUCKET, s3_path)
        else:
            text= extract_text_single(BUCKET, s3_path)
        chunk=chunker(param["chunk"], text)
        domain=create_os_index(param, chunk )
        
def load_document_batch_summary(file_bytes, doc_name): 
    s3_path=f"file_store/{doc_name}"
    S3.put_object(Body=file_bytes,Bucket= BUCKET, Key=s3_path)
    time.sleep(1)    
    text= extract_text(BUCKET, s3_path)
    return text

@st.cache_data
def extract_text_single(file):  
    
    response = TEXTRACT.detect_document_text(
        Document={      
            'Bytes': file,      
        }
    )
    blocks = response['Blocks']   

    bbox=[]
    word=[]
    for item in response['Blocks']:
        if item["BlockType"] == "WORD"  :
            bbox.append(item['Geometry']['BoundingBox'])
            word.append(item["Text"])    
    
    result=" ".join([x for x in word])   
    return result,bbox

def similarity_search(payload, param): 
    """ Function to run similarity search against OpenSearch index"""
    if "titan" in param["emb"].lower():
        prompt= {
            "inputText": payload
            }

        body=json.dumps(prompt)
        modelId = param["emb_model"]
        accept = 'application/json'
        contentType = 'application/json'

        response = BEDROCK.invoke_model(body=body, modelId=modelId, accept=accept,contentType=contentType)
        response_body = json.loads(response.get('body').read())
        embedding=response_body['embedding']
    elif "cohere" in param["emb"].lower(): 
            prompt= {
                "texts": [payload],
             "input_type": "search_document"
            }
            body=json.dumps(prompt)
            modelId = param["emb_model"]
            accept = 'application/json'
            contentType = 'application/json'

            response = BEDROCK.invoke_model(body=body, modelId=modelId, accept=accept,contentType=contentType)
            response_body = json.loads(response.get('body').read())
            embedding=response_body['embeddings'][0]
    else:

        payload = {'text_inputs': payload}
        payload = json.dumps(payload).encode('utf-8')

        response = SAGEMAKER.invoke_endpoint(EndpointName=param["emb_model"], 
                                                    ContentType='application/json',  
                                                    Body=payload)

        model_predictions = json.loads(response['Body'].read())
        embedding = model_predictions['embedding'][0]
    
    query = {
    'size': param["K"],
    'query': {
        "knn": {
          "embedding": {
            "vector": embedding,
            "k": param["knn"]
          }
        }
      }
    }
    

    service = 'es'
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, REGION, service, session_token=credentials.token)
    os_ = OpenSearch(
        hosts = [{'host': OS_ENDPOINT, 'port': 443}],
        http_auth = awsauth,
        use_ssl = True,
        verify_certs = True,
        timeout=120,
        # http_compress = True, # enables gzip compression for request bodies
        connection_class = RequestsHttpConnection
    )
    domain_index=f"{param['domain']}_{param['engine']}"
    response = os_.search(index=domain_index, body=query)
    hits = response['hits']['hits']    
    return hits

@st.cache_data
def extract_text(bucket, filepath):  
    """Function to call textract asynchrounous document text extraction"""
    st.write('Extracting Text')
    response = TEXTRACT.start_document_text_detection(DocumentLocation={'S3Object': {'Bucket':bucket, 'Name':filepath}},JobTag='Employee')
    maxResults = 1000
    paginationToken = None
    finished = False
    jobId=response['JobId']
    print(jobId)
    total_words=[]   
    dict_words={}   
    
    while finished == False:
        response = None               
        if paginationToken == None:
            response = TEXTRACT.get_document_text_detection(JobId=jobId,
                                                                 MaxResults=maxResults)
        else:
            response = TEXTRACT.get_document_text_detection(JobId=jobId,
                                                                 MaxResults=maxResults,
                                                                 NextToken=paginationToken)
    
        while response['JobStatus'] != 'SUCCEEDED':
            time.sleep(2)     
            if paginationToken == None:
                response =TEXTRACT.get_document_text_detection(JobId=jobId,
                                                                 MaxResults=maxResults)
            else:
                response = TEXTRACT.get_document_text_detection(JobId=jobId,
                                                                     MaxResults=maxResults,
                                                                     NextToken=paginationToken)

        blocks = response['Blocks']   
        pages=set(x['Page'] for x in blocks)

        for p in pages:
            a=[]
            b=[]
            for item in response['Blocks']:
                if item["BlockType"] == "WORD" and item['Page'] ==p :
                    a.append(item['Geometry']['BoundingBox'])
                    b.append(item["Text"])
                    total_words.append(item["Text"])
            c=" ".join([x for x in b])
            try:
                dict_words[p]+= f' {c}'                
            except:
                dict_words[p]=c
        if 'NextToken' in response:
            paginationToken = response['NextToken']
        else:
            finished = True
    total_words=" ".join([x for x in total_words])
    return dict_words


def summarize_section(payload):
    """
    Initial summary of chunks
    """
    models=["claude","llama","cohere","ai21","titan","mistral"]
    chosen_model=[x for x in models if x in payload['model_name'].lower()][0]
    with open(f"{PARENT_TEMPLATE_PATH}/summary/{chosen_model}/{payload['persona'].lower()}.txt","r") as f:
        prompt_template=f.read()    
    prompt =prompt_template.format(doc=payload['prompt'])
    if "claude" in payload['model_name'].lower():
        prompt=f"\n\nHuman:\n{prompt}\n\nAssistant:"
    payload['prompt'] = prompt  
    response,i_t,o_t= query_endpoint(payload, prompt)
    return response     
  
def summarize_final(payload, handler=None):
    """
    Final summary of of all chunks summary
    """
    models=["claude","llama","cohere","ai21","titan", "mistral"]
    chosen_model=[x for x in models if x in payload['model_name'].lower()][0]
    with open(f"{PARENT_TEMPLATE_PATH}/summary/{chosen_model}/{payload['persona'].lower()}.txt","r") as f:
        prompt_template=f.read()    
    prompt =prompt_template.format(doc=payload['prompt'])
    if "claude" in payload['model_name'].lower():
        prompt=f"\n\nHuman:\n{prompt}\n\nAssistant:"
    payload['prompt'] = prompt  
    response,i_t,o_t= query_endpoint(payload, prompt, handler)
    return response 

def split_into_sections(paragraph: str, max_words: int) -> list: 
    """
    For Batch document Summarization.
    Spliting by words, with a buffer of 50.
    """
    buffer_len=50
    sections = []
    curr_sect = []    
    # paragraph=' '.join(paragraph.values())
    for word in paragraph.split():
        if len(curr_sect) + 1 <= max_words:
            curr_sect.append(word)        
        else:            
            sections.append(' '.join(curr_sect))
            buffer=sections[-1].split()[-buffer_len:]
            curr_sect = buffer+[word]
    if curr_sect:
        buffer=sections[-1].split()[-buffer_len:]
        curr_sect=buffer+curr_sect
    sections.extend([' '.join(curr_sect) for curr_sect in ([] if not curr_sect else [curr_sect])]) 
    return sections

def first_summ(params, section):
    """
    Function to call initial summary of chunks.
    this function is run in parallel to the number of chunks
    """
    params['prompt']= section
    summaries = summarize_section(params)    
    return summaries

def sec_chunking(word_length,text, params): 
    sec_partial_summary=[]
    num_chunks=word_length//(2000 if not "mistral" in params['model_name'].lower() else 4500)
    #further chucking of initial summary and summarizing (summary of summary) 
    for i in range(num_chunks): 
        num_summary=len(text.split('##'))//num_chunks
        start = i * num_summary  
        end = (i+1) * num_summary
        if i+1==num_chunks:
            part_summary="##".join([x for x in text.split('##')[start:]])
        else:
            part_summary="##".join([x for x in text.split('##')[start:end]])
        sec_partial_summary.append(part_summary)
    return sec_partial_summary

def summarize_context(params):
    """Function for Batch Document Summary"""
    st.title('Batch Document Summarization') 

    if st.button('Summarize', type="primary"):        
        text=' '.join(st.session_state['text'].values())        
        # Calculate token size of input text per model and decide chunking  
        if "claude" in params['model_name'].lower():
            claude = Anthropic()
            token_length=claude.count_tokens(text)
            chunking_needed=token_length>90000
        elif "llama" in params['model_name'].lower():
            tkn=token_counter('meta-llama/Llama-2-13b-chat-hf')
            token_length=len(tkn.encode(text))
            chunking_needed=token_length>3000
        elif "cohere" in params['model_name'].lower():
            encoding=token_cohere("Cohere/command-nightly")
            token_length=len(encoding.encode(text))
            chunking_needed=token_length>3000
        elif "ai21" in params['model_name'].lower():
            #replace with AI21 tokenizer
            encoding = tiktoken.get_encoding('cl100k_base')
            token_length=len(encoding.encode(text))
            chunking_needed=token_length>6000
        elif "titan" in params['model_name'].lower():
            encoding = tiktoken.get_encoding('cl100k_base')
            token_length=len(encoding.encode(text))
            chunking_needed=token_length>6000
        elif "mistral" in params['model_name'].lower():
            tkn=token_counter("mistralai/Mistral-7B-v0.1")
            token_length=len(tkn.encode(text))
            chunking_needed=token_length>7000
        tic = time.time()        
        container_summ=st.empty()
   
        if chunking_needed:
            container_summ.write("Chunking documents...")
            params['max_len']=round(params['first_token_length'])
            sections = split_into_sections(text, params['chunk_len'])  
            num_concurrent_invocations = 10 #number of parallel calls
            pool = multiprocessing.Pool(processes=num_concurrent_invocations)
            results = pool.starmap(first_summ, [(params, section) for section in sections])
            pool.close()
            pool.join() 
            text = "##".join(results) 
            
            word_length=len(text.split())     
            if word_length>max(params['chunk_len']*3,500) if "mistral" in params['model_name'].lower() else max(params['chunk_len']*3,2000): # Check if further chunking is need if text is greater than 3000 words
                sec_partial_summary=sec_chunking(word_length,text,params)
                params['max_len']=round(params['first_token_length'])
                pool = multiprocessing.Pool(processes=num_concurrent_invocations)              
                final_results = pool.starmap(first_summ, [(params, summ) for summ in sec_partial_summary]) 
                pool.close()
                pool.join()   
                #Final summary
                full_summary=[] 
                full_summary="##".join([x for x in final_results])
                params['prompt']=full_summary
                params['max_len']=round(params['second_token_length'])
                summary=summarize_final(params,container_summ)
            else:
                ## Final Summary
                params['prompt']=text
                params['max_len']=round(params['second_token_length'])
                summary=summarize_final(params,container_summ)
        else:
            ## No Chunking Needed Summary
            params['prompt']=text
            params['max_len']=round(params['second_token_length'])
            summary=summarize_final(params,container_summ)
        toc = time.time()
        st.session_state['elapsed'] = round(toc-tic, 1)
        ## Create pdf of summary text
        doc = aw.Document()
        builder = aw.DocumentBuilder(doc) 
        builder.write(summary)
        # Save Locally and upload to s3
        file_name=f"{st.session_state['file_name'].split('.')[0]}_summary.pdf"
        doc.save(file_name, aw.SaveFormat.PDF)
        S3.upload_file(file_name, BUCKET, f"Summary/{file_name}")
        summary+=f"\n\n Link to pdf [summary](https://{BUCKET}.s3.amazonaws.com/Summary/{file_name})"
        st.session_state['summary'] = summary 
        st.subheader(f'Summary (in {st.session_state["elapsed"]} seconds):')
        container_summ.markdown(st.session_state['summary'].replace("$","USD ").replace("%", " percent"))
     
        
        
def summarize_sect(param, payload, handler=None):
    """
    Summary function for Document Insights Action
    """
    models=["claude","llama","cohere","ai21","titan","mistral"]
    chosen_model=[x for x in models if x in param['model_name'].lower()][0]
    with open(f"{PARENT_TEMPLATE_PATH}/summary/{chosen_model}/{param['persona'].lower()}.txt","r") as f:
        prompt_template=f.read()    
    prompt =prompt_template.format(doc=payload)
    if "claude" in param['model_name'].lower():
        prompt=f"\n\nHuman:\n{prompt}\n\nAssistant:" 
  
    param['prompt'] = prompt
    response = query_endpoint(param, prompt, handler)
    return response  

def extract_entities(text, entities):
    """Comprehend extract text entities"""
    extracted_data = {}

    for entity in entities:
        entity_type = entity['Type']
        begin_offset = entity['BeginOffset']
        end_offset = entity['EndOffset']

        if entity_type not in extracted_data:
            extracted_data[entity_type] = []
        extracted_data[entity_type].append(text[begin_offset:end_offset])

    return extracted_data

def page_summary(pdf_file,params):
    """
    This action takes the entire rendered document page as context for the following LLM actions below.
    """
    # Page navigation buttons
    pdf_file=pdf_file.read()
    if pdf_file:
        pdf_document = fitz.open("pdf",pdf_file)    
        page_count=pdf_document.page_count

        colm1,colm2=st.columns([1,1])
        with colm1:
            col1, col2, col3 = st.columns(3)
            # Buttons
            if col1.button("Previous Page", key="prev_page"):
                st.session_state.page_slider-=1        
                st.session_state['page_summ']=""
            if col3.button("Next Page", key="next_page"):
                st.session_state.page_slider+=1
                st.session_state['page_summ']=""
            # Page slider
            if col2.slider("Page Slider", min_value=0, max_value=page_count-1, key="page_slider"): 
                st.session_state['page_summ']=""              
            # Rendering pdf page
            page = pdf_document.load_page(st.session_state.page_slider)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png", 100))) 
            st.image(img)
        with colm2:            
            tab1, tab2, tab3 = st.tabs(["**Entity Extraction**", "**Page Summary**", "**Page QnA**"])
            with tab1:
                c1,c2=st.columns(2)
                ## Uses Amazon Comprehend to detect Entities and PII in the current doc page
                if c1.button('Extract Entities', type="primary", key='entities'): 
                    image_b=pix.tobytes("png", 100)
                    text, bbox=extract_text_single(image_b)           
                    response = COMPREHEND.detect_entities(
                        Text=text,
                        LanguageCode='en',                        
                    )
                    text_container = st.container()
                    entities=[x['Text'] for x in response['Entities']]
                    with text_container:
                        cl1,cl2,cl3=st.columns(3)
                        column=[cl1,cl2,cl3]#,cl6,cl7,cl8,cl9]
                        for x,y in enumerate(entities):
                            index = x%len(column)                        
                            column[index].button(y, key=str(uuid.uuid4()),use_container_width=True)
                            
                if c2.button('Extract PII Entities', type="primary", key='texterd'): 
                    image_b=pix.tobytes("png", 100)
                    text, bbox=extract_text_single(image_b)
                    entities=COMPREHEND.detect_pii_entities(
                        Text=text,
                        LanguageCode='en'
                    )
                    pii=extract_entities(text, entities['Entities'])
                    st.write(pii)

            with tab2:      
                ## Summarize current doc page
                container_tab=st.empty()
                if st.button('Summarize',type="primary",key='summ'): 
                    image_b=pix.tobytes("png", 100)
                    text, bbox=extract_text_single(image_b)
                    summary, input_tok, output_tok=summarize_sect(params, text, container_tab)                
                    st.session_state['page_summ']=summary
                container_tab.markdown(st.session_state['page_summ'].replace("$","USD ").replace("%", " percent"))

            with tab3:
                ## Chat with the current doc page
                response_container = st.container()
                container = st.container()
                image_b=pix.tobytes("png", 100)
                text, bbox=extract_text_single(image_b)
                persona_dict={"General":"assistant","Finance":"finanacial analyst","Insurance":"insurance analyst","Medical":"medical expert"}
                               
                    
                with container:
                    with st.form(key="chat_users", clear_on_submit=True):
                        user_input = st.text_area("You:", key="user_chatter", height=100)
                        submit_button = st.form_submit_button(label='Send')               
     
                    if params["memory"]:
                        user_input=llm_memory(user_input, params=None)
                        
                    if submit_button and user_input:
                        models=["claude","llama","cohere","ai21","titan","mistral"]
                        chosen_model=[x for x in models if x in params['model_name'].lower()][0]
                        with open(f"{PARENT_TEMPLATE_PATH}/rag/{chosen_model}/prompt1.txt","r") as f:
                            prompt_template=f.read()    
                        prompt =prompt_template.format(doc=text, prompt=user_input, role=persona_dict[params['persona']])
                        if "claude" in params['model_name'].lower():
                            prompt=f"\n\nHuman:\n{prompt}\n\nAssistant:"
                        output_answer, input_tok, output_tok=query_endpoint(params,prompt)                   
                        if params["memory"]:
                            chat_history={"user" :user_input,
                            "assistant":output_answer}           
                            if DYNAMODB_TABLE:
                                put_db(chat_history)
                            else:
                                st.session_state['chat_memory'].append(chat_history)

                        st.session_state['message'].append({"\nAnswer": output_answer})
                        st.session_state['past'].append(user_input)             
                        st.session_state['generate'].append(json.dumps(output_answer))

                if st.session_state['generate']:
                    with response_container:
                        for i in range(len(st.session_state['generate'])):
                            message(st.session_state["past"][i].replace("$","USD ").replace("%", " percent"), is_user=True,key=str(uuid.uuid4()))
                            output_answer=json.loads(st.session_state["generate"][i])              
                            message(output_answer.replace("$","USD ").replace("%", " percent"),key=str(uuid.uuid4()))

        
def action_doc(params):   
    st.title('Ask Questions of your Document')
    for message in st.session_state.messages:
        if "role" in message.keys():
            with st.chat_message(message["role"]):            
                st.markdown(message["content"].replace("$","USD ").replace("%", " percent"))
                if "steps" in message.keys():
                    with st.expander(label="**Metadata**"):
                        st.markdown(message["steps"])
        elif "step" in message.keys():
            with st.expander(label="**Additional Response**"):
                st.markdown(message["step"])
                    
    if prompt := st.chat_input(""):  
        if params["memory"]:
            prompt=llm_memory(prompt, params=None) 
        with st.chat_message("user"):
            st.markdown(prompt)       
        st.session_state.messages.append({"role": "user", "content": prompt})

        if round(params["K"])>1 and params['method']=="single-passages":
            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                if "Kendra" in params["rag"]:
                    response=query_index(prompt)  
                elif "OpenSearch" in params["rag"]:            
                    response=similarity_search(prompt, params) 
                output_answer=doc_qna_endpoint(params['endpoint-llm'], response, prompt,params,message_placeholder)
                message_placeholder.markdown(output_answer[0]['Answer'].replace("$","USD ").replace("%", " percent"))
                with st.expander(label="**Metadata**"):             
                    steps=f"""
- **Source:** [1]({output_answer[0]['Link']})
- **Name:** {output_answer[0]['Name']} 
- **Page:** {output_answer[0]['Page']} 
- **Confidence:** {output_answer[0]['Score']}  
- **Input Token:** {output_answer[0]['Input Token']}
- **Output Token:** {output_answer[0]['Output Token']}
- **Model:** {params['model_name']}
"""   
                    st.markdown(steps)            
            
            st.session_state.messages.append({"role": "assistant", "content": output_answer[0]['Answer'], "steps":steps})
            with st.expander(label="**Additional Response**"):     
                for k in range(1, round(params["K"])):
                    steps=f"""
- **Answer {k+1}**: {output_answer[k]['Answer'].replace("$","USD ").replace("%", " percent")}
- **Source:** [{k+1}]({output_answer[k]['Link']})
- **Name:** {output_answer[k]['Name']} 
- **Page:** {output_answer[k]['Page']} 
- **Confidence:** {output_answer[k]['Score']}
- **Input Token:** {output_answer[0]['Input Token']}
- **Output Token:** {output_answer[0]['Output Token']}
- **Model:** {params['model_name']}
"""  
                    st.markdown(steps)            
                    st.session_state.messages.append({"step": steps})
        else:    
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
   
                if "Kendra" in params["rag"]:
                    response=query_index(prompt)  
                elif "OpenSearch" in params["rag"]:            
                    response=similarity_search(prompt, params)   

                output_answer=doc_qna_endpoint(params['endpoint-llm'], response, prompt,params,message_placeholder)
                message_placeholder.markdown(output_answer[0]['Answer'].replace("$","USD ").replace("%", " percent"))
                if 'single-passages' in params['method']:
                    links= f"[1]({output_answer[0]['Link']})"
                else:
                    if isinstance(output_answer[0]['Link'], list):
                        links=""
                        for x,key in enumerate(output_answer[0]['Link']):
                            # Generate a step for each key and add it to the steps string
                            link = f"[{x+1}]({key}), "
                            links += link  
                    else:
                        links= f"[1]({output_answer[0]['Link']})"
                
                
                with st.expander(label="**Metadata**"):             
                    steps=f"""
- **Source:** {links}
- **Name:** {output_answer[0]['Name']} 
- **Page:** {output_answer[0]['Page']} 
- **Confidence:** {output_answer[0]['Score']}
- **Input Token:** {output_answer[0]['Input Token']}
- **Output Token:** {output_answer[0]['Output Token']}
- **Model:** {params['model_name']}
"""   
                    st.markdown(steps)            
                st.session_state.messages.append({"role": "assistant", "content": output_answer[0]['Answer'], "steps":steps})


def app_sidebar():
    with st.sidebar:               
        description = """### AI tool powered by suite of AWS services"""
        st.write(description)
        st.text_input('Total Token Used', str(st.session_state['token'])) 
        st.write('---')
        st.write('### User Preference')
        filepath=None
        file = st.file_uploader('Upload a PDF file', type=['pdf'])       
        
        persona = st.selectbox('Select Persona', ["General","Finance","Insurance","Medical"])
        action_name = st.selectbox('Choose Activity', options=['Document Query', 'Document Insights','Batch Document Summary'])
        llm_model_name = st.selectbox('Select LL Model', options=MODELS_LLM.keys())
        st.session_state['action_name']=action_name
        
        if 'Batch Document Summary' in action_name:           
            top_p = st.slider('Top p', min_value=0., max_value=1., value=1., step=.01)
            temp = st.slider('Temperature', min_value=0., max_value=1., value=0.01, step=.01,  help="Control the response creativity of the LLM, lower values give more conservative answers")
            first_token_length = st.slider('Initial Summary Length', min_value=100., max_value=1000., value=250., step=50., help="Control output token size of each initial document chunk summary" )
            second_token_length = st.slider('Final Summary Length', min_value=100., max_value=10000., value=500., step=100. ,help="Control output token size of final document summary")
            chunk_len = st.slider('Chunk Length', min_value=500., max_value=10000., value=1000., step=500., help="Control amount of words to chunk document into")
            params = {'action_name':action_name,'endpoint-llm':MODELS_LLM[llm_model_name],'first_token_length':first_token_length, 'top_p':top_p, 'temp':temp,
                  'model_name':llm_model_name,'second_token_length':second_token_length, 'chunk_len':chunk_len, "persona":persona }   
            
            if file is not None:
                file_name=str(file.name)
                st.session_state['file_name']=file_name
                st.session_state.generated.append(1) 
                text=load_document_batch_summary(file.read(), file_name)
                st.session_state['text'] =text
                
        if 'Document Insights' in action_name:    
            mem = st.checkbox('chat memory')
            max_len = st.slider('Output Length', min_value=50, max_value=2000, value=250, step=10)
            top_p = st.slider('Top p', min_value=0., max_value=1., value=1., step=.01)
            temp = st.slider('Temperature', min_value=0., max_value=1., value=0.01, step=.01)
            params = {'action_name':action_name, 'endpoint-llm':MODELS_LLM[llm_model_name],'max_len':max_len, 'top_p':top_p, 'temp':temp, 'model_name':llm_model_name, "persona":persona ,"memory":mem }   
            

            
            if file is not None:
                file_name=str(file.name)
                st.session_state['file_name']=file_name
                st.session_state.generated.append(1)
                st.session_state.bytes=file
            

        elif 'Document Query' in action_name: 
            mem = st.checkbox('chat memory')
            
            auto_rtv=st.checkbox('Auto Retrieval Technique')
            if auto_rtv:
                methods=""
            else:
                methods=st.selectbox('retrieval technique', ["single-passages",'combined-passages','full-pages'])
            st.session_state['rtv']=methods
            max_len = st.slider('Output Length', min_value=50, max_value=2000, value=150, step=10)
            top_p = st.slider('Top p', min_value=0., max_value=1., value=1., step=.01)
            temp = st.slider('Temperature', min_value=0., max_value=1., value=0.01, step=.01)
            
            retriever = st.selectbox('Retriever', ("Kendra", "OpenSearch"))
            if auto_rtv:
                K=0
            else:
                K=st.slider('Top K Results', min_value=1., max_value=10., value=1., step=1.,key='kendra')

            if "OpenSearch" in retriever:
                embedding_model=st.selectbox('Embedding Model', MODELS_EMB.keys())
                knn=st.slider('Query Nearest Neighbour', min_value=1., max_value=100., value=3., step=1.)
                # K=st.slider('Top K results', min_value=1., max_value=10., value=1., step=1.)
                engine=st.selectbox('KNN algorithm', ("nmslib", "lucene"), help="Underlying KNN algorithm implementation to use for powering the KNN search")
                m=st.slider('Neighbouring Points', min_value=16.0, max_value=124.0, value=72.0, step=1., help="Explored neighbors count")
                ef_search=st.slider('efSearch', min_value=10.0, max_value=2000.0, value=1000.0, step=10., help="Exploration Factor")
                ef_construction=st.slider('efConstruction', min_value=100.0, max_value=2000.0, value=1000.0, step=10., help="Explain Factor Construction")            
                chunk=st.slider('Word Chunk size', min_value=50, max_value=5000 if "titan" in embedding_model.lower() else 300, value=1000 if "titan" in embedding_model.lower() else 300, step=50,help="Word size to chunk documents into Vector DB") 
                st.session_state['domain']=EMB_MODEL_DOMAIN_NAME[embedding_model.lower()]
                
                params = {'action_name':action_name, 'endpoint-llm':MODELS_LLM[llm_model_name],'max_len':max_len, 'top_p':top_p, 'temp':temp, 
                          'model_name':llm_model_name, "emb_model":MODELS_EMB[embedding_model], "rag":retriever,"K":K, "engine":engine, "m":m,
                         "ef_search":ef_search, "ef_construction":ef_construction, "chunk":chunk, "domain":st.session_state['domain'], "knn":knn,
                         'emb':embedding_model, 'method':methods, "persona":persona,"memory":mem,"auto_rtv":auto_rtv }   
          
            else:
                
                params = {'action_name':action_name, 'endpoint-llm':MODELS_LLM[llm_model_name],"K":K,'max_len':max_len, 'top_p':top_p, 'temp':temp, 'model_name':llm_model_name, "rag":retriever, 'method':methods, "persona":persona,"memory":mem ,"auto_rtv":auto_rtv }   
                
           
            if file is not None:
                file_name=str(file.name)
                st.session_state['file_name']=file_name
                st.session_state.generated.append(1)
                domain=load_document(file, file_name,params) 
      
        return params, file


def main():
    params,f = app_sidebar()
    if params['action_name'] =='Batch Document Summary':
        summarize_context(params) 
    elif params['action_name'] =='Document Insights':
        if st.session_state.bytes:
            page_summary(st.session_state.bytes,params)
    elif params['action_name'] =='Document Query':
        action_doc(params)


if __name__ == '__main__':
    main()
  
