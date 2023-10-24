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
import subprocess
import requests
from requests.auth import HTTPBasicAuth
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
from tokenizers import Tokenizer
from transformers import AutoTokenizer

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
OS_KEY = APP_MD['opensearch']['es_password']
OS_USERNAME =  APP_MD['opensearch']['es_username']
OS_ENDPOINT  =  APP_MD['opensearch']['domain_endpoint']
KENDRA_ID = APP_MD['Kendra']['index']
KENDRA_ROLE=APP_MD['Kendra']['role']
PARENT_TEMPLATE_PATH="prompt_template"
KENDRA_S3_DATA_SOURCE_NAME=APP_MD['Kendra']['s3_data_source_name']


S3            = boto3.client('s3', region_name=REGION)
TEXTRACT      = boto3.client('textract', region_name=REGION)
KENDRA        = boto3.client('kendra', region_name=REGION)
SAGEMAKER     = boto3.client('sagemaker-runtime', region_name=REGION)
BEDROCK = boto3.client(service_name='bedrock-runtime',region_name='us-east-1') 


EMB_MODEL_DICT={"titan":1536,
                "minilmv2":384,
                "bgelarge":1024,
                "gtelarge":1024,
                "e5largev2":1024,
                "e5largemultilingual":1024,
               "gptj6b":4096}

EMB_MODEL_DOMAIN_NAME={"titan":f"{APP_MD['opensearch']['domain_name']}_titan",
                "minilmv2":f"{APP_MD['opensearch']['domain_name']}_minilm",
                "bgelarge":f"{APP_MD['opensearch']['domain_name']}_bgelarge",
                "gtelarge":f"{APP_MD['opensearch']['domain_name']}_gtelarge",
                "e5largev2":f"{APP_MD['opensearch']['domain_name']}_e5large",
                "e5largemultilingual":f"{APP_MD['opensearch']['domain_name']}_e5largeml",
               "gptj6b":f"{APP_MD['opensearch']['domain_name']}_gptj6b"}





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

def chunk_iterator(dir_path: str):
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    file_contents = file.read()
                    yield filename, file_contents


def create_os_index(param, file_location):
    st.write("Indexing...")    
    es_username = OS_USERNAME
    es_password = OS_KEY 
    domain_endpoint = OS_ENDPOINT
    domain_index = param['domain']
    URL = f'{domain_endpoint}/{domain_index}'
    
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
    response = requests.head(URL, auth=HTTPBasicAuth(es_username, es_password))  
    if response.status_code == 404:
        response = requests.put(URL, auth=HTTPBasicAuth(es_username, es_password), json=mapping)
        st.write(f'Index created: {response.text}')
    else:
        st.write('Index already exists!')
        
    i = 1
    for chunk_name, chunk in tqdm(chunk_iterator(file_location)):
        doc_id, chunk_id = chunk_name.split('_',1)
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
        response = requests.post(f'{URL}/_doc/{i}', auth=HTTPBasicAuth(es_username, es_password), json=document)
        i += 1
        if response.status_code not in [200, 201]:
            logger.error(response.status_code)
            logger.error(response.text)
            break
    return domain_index

def split_doc(doc_name):    
    dir_name=doc_name.split('.')[0]
    inputpdf = PdfReader(open(doc_name, "rb"))
    Path(dir_name).mkdir(parents=True, exist_ok=True) 
    for i in range(len(inputpdf.pages)):
        output = PdfWriter()
        output.add_page(inputpdf.pages[i])
        with open(f"{dir_name}/{dir_name}-{i+1}.pdf" ,"wb") as outputStream:
            output.write(outputStream)    
    return dir_name

def kendra_index(doc_name):
    import time
    response=KENDRA.list_data_sources(IndexId=KENDRA_ID)['SummaryItems']
    data_sources=[x["Name"] for x in response if KENDRA_S3_DATA_SOURCE_NAME in x["Name"]]
    if data_sources:
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
    else:
        
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
            # For this example, there should be one job        
            try:
                status = jobs["History"][0]["Status"]
                st.write(" Syncing data source. Status: "+status)
                if status != "SYNCING":
                    status=False
                time.sleep(2)
            except:
                time.sleep(2)
            
def get_chunk_pages(page_dict,chunk):
    encoding = tiktoken.get_encoding('cl100k_base')
    token_dict={}
    length=0
    for pages in page_dict.keys():  
        length+=len(encoding.encode(page_dict[pages]))
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

def chunker(file_name, chunk_size, file):
    chunk_size=round(chunk_size)
    encoding = tiktoken.get_encoding('cl100k_base')
    # file_name=file_name
    doc_id=file_name.split('/')[-1]
    text=' '.join(file.values())
    tokens = encoding.encode(text)
    n_docs = 1
    if not os.path.exists(file_name):
      # Directory does not exist, create it
      os.makedirs(file_name)
    chunk_pages=get_chunk_pages(file,chunk_size)
    for i in range(0, len(tokens), chunk_size): 
        #print(i)
        chunk_tokens = tokens[i: i+chunk_size]   
        chunk = encoding.decode(chunk_tokens)
        with open(f'{file_name}/{doc_id}_{chunk_pages[(int(i)//chunk_size)+1]}', 'w') as f:
            f.write(chunk)
        n_docs += 1    
    st.write(file_name)
    return file_name

def extract_text_single1(file):  
    bucket=file.split('/',4)[3]
    key=file.split('/',4)[-1]
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
            bbox.append(item['Geometry']['BoundingBox'])
            word.append(item["Text"])    

    result=" ".join([x for x in word])
    return result

def doc_qna_endpoint(endpoint, responses,prompt,params):
    models=["claude","llama","cohere","ai21","titan"]
    chosen_model=[x for x in models if x in params['model_name'].lower()][0]
    total_response=[]   

    persona_dict={"General":"assistant","Finance":"finanacial analyst","Insurance":"insurance analyst","Medical":"medical expert"}
    
    if "Kendra" in params["rag"]:
        if 'multiple-passages' in params['method']:
            score = ", ".join([x['ScoreAttributes']["ScoreConfidence"]   for x in responses['ResultItems'][:round(params['K'])]]) 
            doc_link = [x['DocumentURI'] for x in responses['ResultItems'][:round(params['K'])]]
            page_no = ", ".join([x.split('/')[-1].split('-')[-1].split('.')[0] for x in doc_link])      
            
            holder={}
            for x,y in enumerate(responses['ResultItems'][:round(params['K'])]):
                holder[x]={"document":y["Content"], "source":y['DocumentURI']}  
            
            with open(f"{PARENT_TEMPLATE_PATH}/rag/{chosen_model}/prompt1.txt","r") as f:
                prompt_template=f.read()    
            qa_prompt =prompt_template.format(doc=holder, prompt=prompt, role=persona_dict[params['persona']])
            if "claude" in params['model_name'].lower():
                qa_prompt=f"\n\nHuman:\n{qa_prompt}\n\nAssistant:"              
            answer, in_token, out_token=query_endpoint(params, qa_prompt)
            answer1={'Answer':answer, 'Link':doc_link, 'Score':score, 'Page': page_no, "Input Token":in_token,"Output Token":out_token}
            total_response.append(answer1)
            return total_response
        elif 'multiple-full-pages' in params['method']:            
            score = ", ".join([x['ScoreAttributes']["ScoreConfidence"]   for x in responses['ResultItems'][:round(params['K'])]]) 
            doc_link = [x['DocumentURI'] for x in responses['ResultItems'][:round(params['K'])]]
            page_no = ", ".join([x.split('/')[-1].split('-')[-1].split('.')[0] for x in doc_link])  
            holder={}
            for x,y in enumerate(responses['ResultItems']):
                holder[x]={"document":y["Content"], "source":y['DocumentURI']}
            sources=[x['source'] for x in list(holder.values())[:round(params['K'])]]
            import multiprocessing    
            num_concurrent_invocations = len(sources)
            pool = multiprocessing.Pool(processes=num_concurrent_invocations)            
            context=pool.map(extract_text_single1, sources)
            pool.close()
            pool.join() 
            
            with open(f"{PARENT_TEMPLATE_PATH}/rag/{chosen_model}/prompt1.txt","r") as f:
                prompt_template=f.read()    
            qa_prompt =prompt_template.format(doc=context, prompt=prompt, role=persona_dict[params['persona']])
            if "claude" in params['model_name'].lower():
                qa_prompt=f"\n\nHuman:\n{qa_prompt}\n\nAssistant:"  
            
            answer, in_token, out_token=query_endpoint(params, qa_prompt)
            answer1={'Answer':answer, 'Link':doc_link, 'Score':score, 'Page': page_no, "Input Token":in_token,"Output Token":out_token}
            total_response.append(answer1)
            return total_response
        elif 'single-passages' in params['method']:

            with open(f"{PARENT_TEMPLATE_PATH}/rag/{chosen_model}/prompt1_1.txt","r") as f:
                    prompt_template=f.read()
            for x in range(0, round(params['K'])): 
      
                score = responses['ResultItems'][x]['ScoreAttributes']["ScoreConfidence"]              
                passage = responses['ResultItems'][x]['Content']
                doc_link = responses['ResultItems'][x]['DocumentURI']
                page_no = responses['ResultItems'][x]['DocumentURI'].split('/')[-1].split('-')[-1].split('.')[0]
                s3_uri=responses['ResultItems'][x]['DocumentId']
                    
                qa_prompt =prompt_template.format(doc=passage, prompt=prompt, role=persona_dict[params['persona']])
                if "claude" in params['model_name'].lower():
                    qa_prompt=f"\n\nHuman:\n{qa_prompt}\n\nAssistant:"  
                
                answer, in_token, out_token=query_endpoint(params, qa_prompt)
         
                answer1={'Answer':answer, 'Link':doc_link, 'Score':score, 'Page': page_no, "Input Token":in_token,"Output Token":out_token}
                total_response.append(answer1)
            return total_response
        #passage=get_text_link(s3_uri)
    elif "OpenSearch" in params["rag"]:
        if 'single-passages' in params['method']: 
            with open(f"{PARENT_TEMPLATE_PATH}/rag/{chosen_model}/prompt1_1.txt","r") as f:
                    prompt_template=f.read()
            for response in responses:            
                score = response['_score']
                passage = response['_source']['passage']
                doc_link = f"https://{BUCKET}.s3.amazonaws.com/file_store/{response['_source']['doc_id']}"
                page_no = response['_source']['passage_id']
                
                #https://fairstone.s3.amazonaws.com/docqna/amzn-20221231/amzn-20221231-1.pdf      
                qa_prompt =prompt_template.format(doc=passage, prompt=prompt, role=persona_dict[params['persona']])
                if "claude" in params['model_name'].lower():
                    qa_prompt=f"\n\nHuman:\n{qa_prompt}\n\nAssistant:"  
                answer, in_token, out_token=query_endpoint(params, qa_prompt)
                answer1={'Answer':answer, 'Link':doc_link, 'Score':score, 'Page': page_no,"Input Token":in_token,"Output Token":out_token}
                total_response.append(answer1)
                st.write(total_response)
            return total_response
        elif 'multiple-passages' in params['method']:
            with open(f"{PARENT_TEMPLATE_PATH}/rag/{chosen_model}/prompt1.txt","r") as f:
                    prompt_template=f.read()

            score = ",".join([str(x['_score']) for x in responses])
            passage = [x['_source']['passage'] for x in responses]
            doc_link = f"https://{BUCKET}.s3.amazonaws.com/file_store/{responses[0]['_source']['doc_id']}"
            page_no = ",".join([x['_source']['passage_id'] for x in responses])
            #https://fairstone.s3.amazonaws.com/docqna/amzn-20221231/amzn-20221231-1.pdf      
            qa_prompt =prompt_template.format(doc=passage, prompt=prompt, role=persona_dict[params['persona']])
            if "claude" in params['model_name'].lower():
                qa_prompt=f"\n\nHuman:\n{qa_prompt}\n\nAssistant:"  
            answer, in_token, out_token=query_endpoint(params, qa_prompt)
            answer1={'Answer':answer, 'Link':doc_link, 'Score':score, 'Page': page_no,"Input Token":in_token,"Output Token":out_token}
            total_response.append(answer1)
            return total_response
          
            
            
def query_endpoint(params, qa_prompt):
    if 'ai21' in params['model_name'].lower():

        import json        
        prompt={
          "prompt":  qa_prompt,
          "maxTokens": params['max_len'],
          "temperature": round(params['temp'],2),
          # "topP":  params['top_p'], 
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
          # "top_k": 50,
          # "top_p": params['top_p'],  
        }
        prompt=json.dumps(prompt)
        response = BEDROCK.invoke_model(body=prompt,
                                modelId=params['endpoint-llm'],  
                                accept="application/json", 
                                contentType="application/json")
        answer=response['body'].read().decode()
        answer=json.loads(answer)['completion']
        claude = Anthropic()
        input_token=claude.count_tokens(qa_prompt)
        output_token=claude.count_tokens(answer)
        tokens=claude.count_tokens(f"{qa_prompt} {answer}")
        st.session_state['token']+=tokens

    elif 'titan' in params['model_name'].lower():
      
        import json
        encoding = tiktoken.get_encoding('cl100k_base')        
        prompt={
               "inputText": qa_prompt,
               "textGenerationConfig": {
                   "maxTokenCount": params['max_len'],     
                   "temperature":params['temp'],
                   "topP":params['top_p'],  
                   },
            }
        prompt=json.dumps(prompt)
        response = BEDROCK.invoke_model(body=prompt,
                                modelId=params['endpoint-llm'], 
                                accept="application/json", 
                                contentType="application/json")
        answer=response['body'].read().decode()
        answer=json.loads(answer)['results'][0]['outputText']
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
        response = BEDROCK.invoke_model(body=prompt,
                                modelId=params['endpoint-llm'], 
                                accept="application/json", 
                                contentType="application/json")
        answer=response['body'].read().decode()
        answer=json.loads(answer)["generations"][0]['text']
        encoding=token_cohere("Cohere/command-nightly")
        input_token=len(encoding.encode(qa_prompt))
        output_token=len(encoding.encode(answer))
        tokens=len(encoding.encode(f"{qa_prompt} {answer}"))
        st.session_state['token']+=tokens

    elif 'llama2' in params['model_name'].lower(): 

        import boto3
        import json       

        payload = {
           "inputs": qa_prompt,
            "parameters": {"max_new_tokens": params['max_len'], 
                           "top_p": params['top_p'] if params['top_p']<1 else 0.99 ,
                           "temperature": params['temp'] if params['temp']>0 else 0.01,
                           "return_full_text": False,}
        }
        llama=boto3.client("sagemaker-runtime")
        output=llama.invoke_endpoint(Body=json.dumps(payload), EndpointName=params['endpoint-llm'],ContentType="application/json",CustomAttributes='accept_eula=true')
        answer=json.loads(output['Body'].read().decode())[0]['generated_text']
        tkn=token_counter("heilerich/llama-tokenizer-fast")
        input_token=len(tkn.encode(qa_prompt))
        output_token=len(tkn.encode(answer))       
        tokens=len(tkn.encode(f"{qa_prompt} {answer}"))
        st.session_state['token']+=tokens       
        
    return answer, input_token, output_token
        

def load_document(file_bytes, doc_name, param):
    if "Kendra" in param["rag"]:    
        with open(doc_name, 'wb') as fp:
            fp.write(file_bytes)
        inputpdf = PdfReader(open(doc_name, "rb"))
        if len(inputpdf.pages)>1:
            dir_folder=split_doc(doc_name)
            subprocess.run(["aws", "s3", "sync", dir_folder, f"s3://{BUCKET}/{PREFIX}/{dir_folder}/", "--acl", "public-read"])    
            time.sleep(3)
            st.write('Kendra Indexing')
            kendra_index(dir_folder) 
        elif len(inputpdf.pages)==1:
            dir_folder=doc_name.split('.')[0]
            subprocess.run(["aws", "s3", "sync", doc_name, f"s3://{BUCKET}/{PREFIX}/{dir_folder}/", "--acl", "public-read"])  
            st.write('Kendra Indexing')
            kendra_index(dir_folder) 
    elif "OpenSearch" in param["rag"]:
        with open(doc_name, 'wb') as fp:
            fp.write(file_bytes)
        inputpdf = PdfReader(open(doc_name, "rb"))
        if len(inputpdf.pages)>1:
            text, file_dir=load_documents(file_bytes, doc_name)
            chunk_path=chunker(file_dir,param["chunk"], text)
            domain=create_os_index(param, chunk_path)
            return domain
        elif len(inputpdf.pages)==1:
            text, file_dir= load_document_single(file_bytes, doc_name)
            chunk_path=chunker(file_dir, param["chunk"], text)
            domain=create_os_index(param, chunk_path)            
            return domain 
        
def load_documents(file_bytes, doc_name): 
    s3_path=f"file_store/{doc_name}"
    with open(doc_name, 'rb') as fp:
        S3.upload_fileobj(fp, BUCKET, s3_path,ExtraArgs={'ACL': 'public-read'})     
    time.sleep(3)
    
    text,file_p= extract_text(BUCKET, s3_path)
    return text,file_p

def load_document_single(file_bytes, doc_name):  
    s3_path=f"file_store/{doc_name}"
    with open(doc_name, 'rb') as fp:
        S3.upload_fileobj(fp, BUCKET, s3_path)
    st.write('Extracting Text')
    text,file_p = extract_text_single(BUCKET, s3_path)
    return text,file_p

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
    es_username = OS_USERNAME
    es_password = OS_KEY 
    domain_endpoint = OS_ENDPOINT
    URL = f'{domain_endpoint}/{param["domain"]}/_search'     
    response = requests.post(URL, auth=HTTPBasicAuth(es_username, es_password), json=query)
    response_json = response.json()
    hits = response_json['hits']['hits']    
    return hits

@st.cache_data
def extract_text(bucket, filepath):   
    st.write('Extracting Text')

    response = TEXTRACT.start_document_text_detection(DocumentLocation={'S3Object': {'Bucket':bucket, 'Name':filepath}},JobTag='Employee')
    maxResults = 1000
    paginationToken = None
    finished = False
    jobId=response['JobId']
    print(jobId)
    total_words=[]
    file=filepath.split('/')[-1].split('.pdf')[0]
    word_prefix=f'words/{file}'
    dict_words={}   
    total_words=[]
    while finished == False:
        response = None               
        if paginationToken == None:
            response = TEXTRACT.get_document_text_detection(JobId=jobId,
                                                                 MaxResults=maxResults)
        else:
            response = TEXTRACT.get_document_text_detection(JobId=jobId,
                                                                 MaxResults=maxResults,
                                                                 NextToken=paginationToken)
        #print(response['JobStatus'])
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
    return dict_words, f'{word_prefix}/{file}'

#function to summarize initial chunks
def summarize_section(payload):

    models=["claude","llama","cohere","ai21","titan"]
    chosen_model=[x for x in models if x in payload['model_name'].lower()][0]
    with open(f"{PARENT_TEMPLATE_PATH}/summary/{chosen_model}/{payload['persona'].lower()}.txt","r") as f:
        prompt_template=f.read()    
    prompt =prompt_template.format(doc=payload['prompt'])
    if "claude" in payload['model_name'].lower():
        prompt=f"\n\nHuman:\n{prompt}\n\nAssistant:"
    payload['prompt'] = prompt  
    response,i_t,o_t= query_endpoint(payload, prompt)
    return response     
  
def summarize_final(payload):
    
    models=["claude","llama","cohere","ai21","titan"]
    chosen_model=[x for x in models if x in payload['model_name'].lower()][0]
    with open(f"{PARENT_TEMPLATE_PATH}/summary/{chosen_model}/{payload['persona'].lower()}.txt","r") as f:
        prompt_template=f.read()    
    prompt =prompt_template.format(doc=payload['prompt'])
    if "claude" in payload['model_name'].lower():
        prompt=f"\n\nHuman:\n{prompt}\n\nAssistant:"
    payload['prompt'] = prompt  
    response,i_t,o_t= query_endpoint(payload, prompt)
    return response 


#function to split extracted text to chunks
def split_into_sections(paragraph: str, max_words: int) -> list: 
    sections = []
    curr_sect = []    
    # paragraph=' '.join(paragraph.values())
    for word in paragraph.split():
        if len(curr_sect) + 1 <= max_words:
            curr_sect.append(word)        
        else:
            if len(sections)<1:
                sections.append(' '.join(curr_sect))
                curr_sect = [word]
            else:
                buffer=sections[-1].split()[-50:]
                curr_sect=buffer+curr_sect
                sections.append(' '.join(curr_sect))
                curr_sect = [word]
    if curr_sect:
        buffer=sections[-1].split()[-50:]
        curr_sect=buffer+curr_sect
    sections.extend([' '.join(curr_sect) for curr_sect in ([] if not curr_sect else [curr_sect])]) 
    return sections

def first_summ(params, section):
    params['prompt']= section
    summaries = summarize_section(params)    
    return summaries

def sec_chunking(char_length,text): 
    sec_partial_summary=[]
    num_chunks=char_length//10000
    #further chucking of initial summary and summarizing (summary of summary) 
    for i in range(num_chunks): 
        num_summary=len(text.split('##\n\n'))//num_chunks
        start = i * num_summary  
        end = (i+1) * num_summary
        if i+1==num_chunks:
            part_summary="\n\n".join([x for x in text.split('##\n\n')[start:]])
        else:
            part_summary="\n\n".join([x for x in text.split('##\n\n')[start:end]])
        sec_partial_summary.append(part_summary)
    return sec_partial_summary

def summarize_context(params):
    st.title('Batch Document Summarization') 
    
    if st.button('Summarize', type="primary"):
        
        text=' '.join(st.session_state['text'].values())
        # Calculate token size of input text per model and decide chunking
        if "claude" in params['model_name'].lower():
            claude = Anthropic()
            token_length=claude.count_tokens(text)
            chunking_needed=token_length>80000
        elif "llama" in params['model_name'].lower():
            tkn=token_counter('meta-llama/Llama-2-13b-chat-hf')
            token_length=len(tkn.encode(text))
            chunking_needed=token_length>3000
        elif "cohere" in params['model_name'].lower():
            encoding=token_cohere("Cohere/command-nightly")
            token_length=len(encoding.encode(text))
            chunking_needed=token_length>3000
        elif "ai21" in params['model_name'].lower():
            #replace with AI21 tokenizer, using number of words instead
            encoding = tiktoken.get_encoding('cl100k_base')
            token_length=len(encoding.encode(text))
            chunking_needed=token_length>6000
        elif "titan" in params['model_name'].lower():
            encoding = tiktoken.get_encoding('cl100k_base')
            token_length=len(encoding.encode(text))
            chunking_needed=token_length>6000
        tic = time.time()
        if chunking_needed:
            st.write("Chunking documents...")
            params['max_len']=round(params['first_token_length'])
            sections = split_into_sections(text, params['chunk_len'])  
            num_concurrent_invocations = 10
            pool = multiprocessing.Pool(processes=num_concurrent_invocations)
            results = pool.starmap(first_summ, [(params, section) for section in sections])
            pool.close()
            pool.join() 
            text = '##\n\n'.join(results) 
            char_length=sum([len(x) for x in text]) 
            if char_length>params['chunk_len']*10: 
                sec_partial_summary=sec_chunking(char_length,text)
                params['max_len']=round(params['first_token_length'])
                pool = multiprocessing.Pool(processes=num_concurrent_invocations)              
                final_results = pool.starmap(first_summ, [(params, summ) for summ in sec_partial_summary])    
                # Close the pool and wait for all processes to finish
                pool.close()
                pool.join()   
                #Final summary
                full_summary=[] 
                full_summary="\n\n".join([x for x in final_results])
                params['prompt']=full_summary
                params['max_len']=round(params['second_token_length'])
                summary=summarize_final(params)
            else:
                params['prompt']=text
                params['max_len']=round(params['second_token_length'])
                summary=summarize_final(params)
        else:
            params['prompt']=text
            params['max_len']=round(params['second_token_length'])
            summary=summarize_final(params)
        toc = time.time()
        st.session_state['elapsed'] = round(toc-tic, 1)
        with open('output.txt','w') as f:
            for line in summary.split('\n'):
                f.write(line)
        # load TXT document
        doc = aw.Document('output.txt')        
        # save TXT as PDF file
        doc.save("txt-2-pdf.pdf", aw.SaveFormat.PDF)
        #upload to s3 and make public. Do take out the "public-read" from the call if you do not want the file to be public
        subprocess.run(["aws", "s3", "cp", "txt-2-pdf.pdf", f"s3://{BUCKET}/{PREFIX}/txt-2-pdf.pdf", "--acl", "public-read"]) 
        summary+=f"\n\n Link to pdf [summary](https://{BUCKET}.s3.amazonaws.com/{PREFIX}/txt-2-pdf.pdf)"
        st.session_state['summary'] = summary     
        st.subheader(f'Summary (in {st.session_state["elapsed"]} seconds):')
        # st.experimental_rerun()
        st.markdown(st.session_state['summary'].replace("$","USD ").replace("%", " percent"))
        
        
def summarize_sect(param, payload):
    models=["claude","llama","cohere","ai21","titan"]
    chosen_model=[x for x in models if x in param['model_name'].lower()][0]
    with open(f"{PARENT_TEMPLATE_PATH}/summary/{chosen_model}/{param['persona'].lower()}.txt","r") as f:
        prompt_template=f.read()    
    prompt =prompt_template.format(doc=payload)
    if "claude" in param['model_name'].lower():
        prompt=f"\n\nHuman:\n{prompt}\n\nAssistant:" 
  
    param['prompt'] = prompt
    response = query_endpoint(param, prompt)
    return response  

def page_summary(pdf_file,params):
    # Page navigation buttons
    pdf_file=pdf_file.read()
    if pdf_file:
        pdf_document = fitz.open("pdf",pdf_file)    
        page_count=pdf_document.page_count

        colm1,colm2=st.columns([1,1])
        with colm1:
            col1, col2, col3 = st.columns(3)
            if col1.button("Previous Page", key="prev_page"):
                st.session_state.current_page-=1        
                st.session_state['page_summ']=""
            if col3.button("Next Page", key="next_page"):
                st.session_state.current_page+=1
                st.session_state['page_summ']=""
            # Page slider
            if col2.slider("Page Slider", min_value=0, max_value=page_count-1, value=st.session_state.current_page, key="page_slider"):
                st.session_state.current_page= st.session_state.page_slider
                st.session_state['page_summ']=""

            page = pdf_document.load_page(st.session_state.current_page)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png", 100))) 
            st.image(img)
        with colm2:
            tab1, tab2, tab3 = st.tabs(["**Textract**", "**Page Summary**", "**Page QnA**"])
            with tab1:
                if st.button('Extract Text', type="primary", key='texterd'): 
                    image_b=pix.tobytes("png", 100)
                    text, bbox=extract_text_single(image_b)
                    text=text.split(' ')
                    text_container = st.container()

                    with text_container:
                        cl1,cl2,cl3,cl4,cl5=st.columns(5)
                        column=[cl1,cl2,cl3,cl4,cl5]#,cl6,cl7,cl8,cl9]
                        for x,y in enumerate(text):
                            index = x%len(column)                        
                            column[index].button(y, key=str(uuid.uuid4()),use_container_width=True)


            with tab2:
                if st.button('Summarize',key='summ'):               
                    image_b=pix.tobytes("png", 100)
                    text, bbox=extract_text_single(image_b)
                    summary, input_tok, output_tok=summarize_sect(params, text)                
                    st.session_state['page_summ']=summary
                st.markdown(st.session_state['page_summ'].replace("$","USD ").replace("%", " percent"))

            with tab3:
                response_container = st.container()
                container = st.container()
                image_b=pix.tobytes("png", 100)
                text, bbox=extract_text_single(image_b)
                persona_dict={"General":"assistant","Finance":"finanacial analyst","Insurance":"insurance analyst","Medical":"medical expert"}
                with container:
                    with st.form(key='my_form', clear_on_submit=True):
                        user_input = st.text_area("You:", key='input', height=100)
                        submit_button = st.form_submit_button(label='Send')
                    # Initialise session state variables
                    
                    if submit_button and user_input:
                        models=["claude","llama","cohere","ai21","titan"]
                        chosen_model=[x for x in models if x in params['model_name'].lower()][0]
                        with open(f"{PARENT_TEMPLATE_PATH}/rag/{chosen_model}/prompt1.txt","r") as f:
                            prompt_template=f.read()    
                        prompt =prompt_template.format(doc=text, prompt=user_input, role=persona_dict[params['persona']])
                        if "claude" in params['model_name'].lower():
                            prompt=f"\n\nHuman:\n{prompt}\n\nAssistant:"
                        output_answer, input_tok, output_tok=query_endpoint(params,prompt)               

                        st.session_state['message'].append({"\nAnswer": output_answer})
                        st.session_state['past'].append(user_input)             
                        st.session_state['generate'].append(json.dumps(output_answer))

                if st.session_state['generate']:
                    with response_container:
                        for i in range(len(st.session_state['generate'])):
                            message(st.session_state["past"][i].replace("$","USD ").replace("%", " percent"), is_user=True, key=str(uuid.uuid4()))
                            output_answer=json.loads(st.session_state["generate"][i])              
                            message(output_answer.replace("$","USD ").replace("%", " percent"))

        
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
                    
    if prompt := st.chat_input("Hello?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        if round(params["K"])>1 and params['method']=="single-passages":# and "opensearch" in params["rag"].lower():
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                if "Kendra" in params["rag"]:
                    response=query_index(prompt)  
                elif "OpenSearch" in params["rag"]:            
                    response=similarity_search(prompt, params)                
                output_answer=doc_qna_endpoint(params['endpoint-llm'], response, prompt,params)
                message_placeholder.markdown(output_answer[0]['Answer'].replace("$","USD ").replace("%", " percent"))
                with st.expander(label="**Metadata**"):             
                    steps=f"""
                    - Source: [1]({output_answer[0]['Link']})
                    - Page: {output_answer[0]['Page']} 
                    - Confidence: {output_answer[0]['Score']}  
                    - Input Token: {output_answer[0]['Input Token']}
                    - Output Token: {output_answer[0]['Output Token']}
                    """   
                    st.markdown(steps)            
            
            st.session_state.messages.append({"role": "assistant", "content": output_answer[0]['Answer'], "steps":steps})
            with st.expander(label="**Additional Response**"):
                # try:
      
                    for k in range(1, round(params["K"])):
                        steps=f"""
                        - **Answer {k+1}**: {output_answer[k]['Answer'].replace("$","USD ").replace("%", " percent")}
                        - **Source:** [{k+1}]({output_answer[k]['Link']})
                        - **Page:** {output_answer[k]['Page']} 
                        - **Confidence:** {output_answer[k]['Score']}
                        - **Input Token:** {output_answer[0]['Input Token']}
                        - **Output Token:** {output_answer[0]['Output Token']}
                        """  
                        st.markdown(steps)            
                        st.session_state.messages.append({"step": steps})
                # except:
                #     pass
            
            
        else:    
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                if "Kendra" in params["rag"]:
                    response=query_index(prompt)  
                elif "OpenSearch" in params["rag"]:            
                    response=similarity_search(prompt, params)   

                output_answer=doc_qna_endpoint(params['endpoint-llm'], response, prompt,params)
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
                    - Source: {links}
                    - Page: {output_answer[0]['Page']} 
                    - Confidence: {output_answer[0]['Score']}
                    - Input Token: {output_answer[0]['Input Token']}
                    - Output Token: {output_answer[0]['Output Token']}
                    """   
                    st.markdown(steps)            
                st.session_state.messages.append({"role": "assistant", "content": output_answer[0]['Answer'], "steps":steps})
        st.rerun()


def app_sidebar():
    with st.sidebar:
        st.text_input('Total Token Used', str(st.session_state['token']))
        st.write('## How to use:')
        description = """Welcome to our LLM tool extraction and query answering application."""
        st.write(description)
        st.write('---')
        st.write('### User Preference')
        filepath=None
        file = st.file_uploader('Upload a PDF file', type=['pdf'])       
        
        persona = st.selectbox('Select Persona', ["General","Finance","Insurance","Medical"])
        action_name = st.selectbox('Choose Activity', options=['Document Query', 'Document Insights','Batch Document Summary'])
        llm_model_name = st.selectbox('Select LL Model', options=MODELS_LLM.keys())
        
        
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
                text, file_dir=load_documents(file.read(), file_name)
                st.session_state['text'] =text
                
        if 'Document Insights' in action_name:           
            max_len = st.slider('Output Length', min_value=50, max_value=2000, value=250, step=10)
            top_p = st.slider('Top p', min_value=0., max_value=1., value=1., step=.01)
            temp = st.slider('Temperature', min_value=0., max_value=1., value=0.01, step=.01)
            params = {'action_name':action_name, 'endpoint-llm':MODELS_LLM[llm_model_name],'max_len':max_len, 'top_p':top_p, 'temp':temp, 'model_name':llm_model_name, "persona":persona }   

            
            if file is not None:
                file_name=str(file.name)
                st.session_state['file_name']=file_name
                st.session_state.generated.append(1)
                st.session_state.bytes=file
            

        elif 'Document Query' in action_name: 
            methods=st.selectbox('retrieval technique', ["single-passages",'multiple-passages','multiple-full-pages'])
            st.session_state['rtv']=methods
            max_len = st.slider('Max Length', min_value=50, max_value=2000, value=150, step=10)
            top_p = st.slider('Top p', min_value=0., max_value=1., value=1., step=.01)
            temp = st.slider('Temperature', min_value=0., max_value=1., value=0.01, step=.01)
            
            retriever = st.selectbox('Retriever', ("Kendra", "OpenSearch"))

            if "OpenSearch" in retriever:
                embedding_model=st.selectbox('Embedding Model', MODELS_EMB.keys())
                knn=st.slider('Query Nearest Neighbour', min_value=1., max_value=100., value=3., step=1.)
                K=st.slider('Top K results', min_value=1., max_value=10., value=1., step=1.)
                engine=st.selectbox('KNN algorithm', ("nmslib", "lucene"), help="Underlying KNN algorithm implementation to use for powering the KNN search")
                m=st.slider('Neighbouring Points', min_value=16.0, max_value=124.0, value=72.0, step=1., help="Explored neighbors count")
                ef_search=st.slider('efSearch', min_value=10.0, max_value=2000.0, value=1000.0, step=10., help="Exploration Factor")
                ef_construction=st.slider('efConstruction', min_value=100.0, max_value=2000.0, value=1000.0, step=10., help="Explain Factor Construction")            
                chunk=st.slider('Token Chunk size', min_value=100.0, max_value=5000.0, value=1000.0, step=100.,help="Token size to chunk documents into Vector DB") 
                st.session_state['domain']=EMB_MODEL_DOMAIN_NAME[embedding_model.lower()]
                
                params = {'action_name':action_name, 'endpoint-llm':MODELS_LLM[llm_model_name],'max_len':max_len, 'top_p':top_p, 'temp':temp, 
                          'model_name':llm_model_name, "emb_model":MODELS_EMB[embedding_model], "rag":retriever,"K":K, "engine":engine, "m":m,
                         "ef_search":ef_search, "ef_construction":ef_construction, "chunk":chunk, "domain":st.session_state['domain'], "knn":knn,
                         'emb':embedding_model, 'method':methods, "persona":persona }   
          
            else:
                K=st.slider('Top K Results', min_value=1., max_value=10., value=1., step=1.,key='kendra')
                params = {'action_name':action_name, 'endpoint-llm':MODELS_LLM[llm_model_name],"K":K,'max_len':max_len, 'top_p':top_p, 'temp':temp, 'model_name':llm_model_name, "rag":retriever, 'method':methods, "persona":persona }   
                
           
            if file is not None:
                file_name=str(file.name)
                st.session_state['file_name']=file_name
                st.session_state.generated.append(1)
                domain=load_document(file.read(), file_name,params) 
      
        return params, file


def main():
    params,f = app_sidebar()
    #endpoint=params['endpoint']
    if params['action_name'] =='Batch Document Summary':
        summarize_context(params) 
    elif params['action_name'] =='Document Insights':
        if st.session_state.bytes:
            page_summary(st.session_state.bytes,params)
    else:
        action_doc(params)


if __name__ == '__main__':
    main()
