import pandas as pd
import sys
import argparse
import os
import json
import pickle
import glob
import time
from datetime import datetime
import openai
from dotenv import load_dotenv
from llama_index import VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding
from llama_index.vector_stores import ChromaVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from llama_index.prompts import Prompt
import warnings
warnings.filterwarnings("ignore")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 32

def initialize_and_return_models():
    # os.environ["OPENAI_API_KEY"] = config_dict['openai_api_key']
    # load_dotenv("openai.env")
    # openai.api_key=os.getenv('OPENAI_API_KEY')
    openai.api_key = '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'
    llm = OpenAI(model='gpt-3.5-turbo', temperature=0.5)
    embedding_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    return llm, embedding_model

def load_target_dfs():
    with open('/home/moningi-srija/Desktop/targets_train_df.pkl', 'rb') as handle:
        df_train = pickle.load(handle)
    with open('/home/moningi-srija/Desktop/targets_test_df.pkl', 'rb') as handle:
        df_test = pickle.load(handle)
    #Convert report_date column to string representation
    df_train['report_date'] = df_train['report_date'].apply(lambda x: x.date().strftime('%Y-%m-%d'))
    df_train.reset_index(drop=True, inplace=True)
    df_test['report_date'] = df_test['report_date'].apply(lambda x: x.date().strftime('%Y-%m-%d'))
    df_test.reset_index(drop=True, inplace=True)
    return df_train, df_test

def get_systemprompt_template(config_dict):
    chat_text_qa_msgs = [
        SystemMessagePromptTemplate.from_template(
            ". Give the answer in json format with only one key that is: 'score'. The value should be between 0 and 100\n"
        ),
        HumanMessagePromptTemplate.from_template(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information, "
            "answer the question: {query_str}\n"
        ),
    ]
    chat_text_qa_msgs_lc = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    text_qa_template = Prompt.from_langchain_prompt(chat_text_qa_msgs_lc)
    # text_qa_template = Prompt.from_prompt_template(chat_text_qa_msgs_lc)
    # text_qa_template = Prompt.from_messages(chat_text_qa_msgs_lc)
    return text_qa_template

def get_gpt_generated_feature_dict(query_engine, questions_dict):
    '''
    Returns:
        A dictionary with keys as question identifiers and value as GPT scores.
    '''
    response_dict = {}
    for feature_name, question in questions_dict.items():
        #Sleep for a short duration, not to exceed openai rate limits.
        # while True:
        #     try:
        #         response = query_engine.query(question)
        #         print("hiii\n")
        #         break
        #     except openai.error.RateLimitError:
        #         print("Rate limit reached. Retrying in 20 seconds...")
        #         time.sleep(20)
        response = query_engine.query(question)
        print("hiii\n")
        response_dict[feature_name] = int(eval(response.response)['score'])
    return response_dict

def load_index(llm, embedding_model, base_embeddings_path, symbol, ar_date):
    '''
    Function to load the embeddings that were saved using embeddings_save.py
    '''
    db = chromadb.PersistentClient(path=os.path.join(base_embeddings_path, symbol, ar_date))
    chroma_collection = db.get_collection("ar_date")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    service_context = ServiceContext.from_defaults(embed_model=embedding_model, 
                                                   llm=llm, 
                                                   chunk_size = CHUNK_SIZE, 
                                                   chunk_overlap=CHUNK_OVERLAP)
    index = VectorStoreIndex.from_vector_store(
                vector_store,
                service_context=service_context,
            )
    return index

def load_query_engine(index, text_qa_template):
    return index.as_query_engine(text_qa_template=text_qa_template)

def are_features_generated(base_path, symbol, ar_date):
    '''
    Function to check if the features df has already been created before.
    '''
    df_name = 'df_{}_{}.pkl'.format(symbol, ar_date)
    full_path = os.path.join(base_path, df_name)
    if os.path.exists(full_path):
        return True
    return False

def save_features(df, llm, embedding_model, config_dict, questions_dict,
                  embeddings_directory, features_save_directory):
    '''
    Function to iteratively save features as a df with single row.
    '''
    for i in df.index:
        start_time = time.time()
        curr_series = df.loc[i]
        symbol = curr_series['symbol']
        ar_date = curr_series['report_date']
        if are_features_generated(features_save_directory, symbol, ar_date):
            continue
        index = load_index(llm, embedding_model, embeddings_directory, symbol, ar_date)
        text_qa_template = get_systemprompt_template(config_dict)
        query_engine = load_query_engine(index, text_qa_template)
        time.sleep(20)
        #Get feature scores as dictionary
        gpt_feature_dict = get_gpt_generated_feature_dict(query_engine, questions_dict)
        #Convert dictionary to dataframe
        gpt_feature_df = pd.DataFrame.from_dict(gpt_feature_dict, orient='index').T
        gpt_feature_df.columns = ['feature_{}'.format(c) for c in gpt_feature_df.columns]
        gpt_feature_df['meta_symbol'] = symbol
        gpt_feature_df['meta_report_date'] = ar_date
        os.makedirs(features_save_directory, exist_ok=True)
        with open(os.path.join(features_save_directory, 'df_{}_{}.pkl'.format(symbol, ar_date)), 'wb') as handle:
            print(gpt_feature_df)
            pickle.dump(gpt_feature_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Completed: {} in {:.2f}s".format(i, time.time()-start_time))

def save_consolidated_df( questions_dict, targets_df,
                         features_save_directory, final_df_save_path):
    df_paths_list = [file for file in glob.glob(os.path.join(features_save_directory, '*')) \
                  if os.path.isfile(file)]
    feature_df_full = pd.DataFrame()
    feature_cols = list(questions_dict.keys())
    feature_cols = ['feature_{}'.format(f) for f in feature_cols]
    meta_cols = ['meta_symbol', 'meta_report_date']
    for df_path in df_paths_list:
        with open(df_path, 'rb') as handle:
            gpt_feature_df = pickle.load(handle)
            print(gpt_feature_df)
        gpt_feature_df = gpt_feature_df.loc[:, feature_cols + meta_cols].copy()
        feature_df_full = pd.concat([feature_df_full, gpt_feature_df], ignore_index=True)
    #Convert meta_report_date column to datetime format
    feature_df_full['meta_report_date'] = feature_df_full['meta_report_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    merged_df = pd.merge(feature_df_full, targets_df, left_on=['meta_symbol', 'meta_report_date'],
                        right_on=['symbol', 'report_date'], how='inner')
    #Transform features in range [0,1]
    merged_df[feature_cols] = merged_df[feature_cols]/100.0
    with open(final_df_save_path, 'wb') as handle:
        pickle.dump(merged_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(args):
    # with open(args.config_path) as json_file:
    #     config_dict = json.load(json_file)
    with open("/home/moningi-srija/Desktop/questions.json") as json_file:
        questions_dict = json.load(json_file)
    
    df_train, df_test = load_target_dfs()
    llm, embedding_model = initialize_and_return_models()

    save_features(df_train, llm, embedding_model, "/home/moningi-srija/Desktop", questions_dict,
                  embeddings_directory="/home/moningi-srija/Desktop/embeddings_for_training.pkl",
                  features_save_directory='media/moningi-srija/Seagate\ Backup\ Plus\ Drive/ai_ml/GPT-InvestAR-main/feature_train3')
    save_features(df_test, llm, embedding_model, '/home/moningi-srija/Desktop', questions_dict,
                  embeddings_directory='/home/moningi-srija/Desktop/embeddings_for_testing.pkl',
                  features_save_directory='media/moningi-srija/Seagate\ Backup\ Plus\ Drive/ai_ml/GPT-InvestAR-main/feature_test3')
    
    save_consolidated_df(questions_dict, df_train,
                         features_save_directory='media/moningi-srija/Seagate\ Backup\ Plus\ Drive/ai_ml/GPT-InvestAR-main/feature_train3',
                         final_df_save_path='media/moningi-srija/Seagate\ Backup\ Plus\ Drive/ai_ml/GPT-InvestAR-main/final_train3')
    save_consolidated_df(questions_dict, df_test,
                         features_save_directory='/media/moningi-srija/Seagate\ Backup\ Plus\ Drive/ai_ml/GPT-InvestAR-main/feature_test3',
                         final_df_save_path='media/moningi-srija/Seagate\ Backup\ Plus\ Drive/ai_ml/GPT-InvestAR-main/final_test3')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_path', dest='config_path', type=str,
    #                     required=True,
    #                     help='''Full path of config.json''')
    # parser.add_argument('--questions_path', dest='questions_path', type=str,
    #                     required=True,
    #                     help='''Full path of questions.json which contains the questions 
    #                     for asking to the LLM''')
    main(args=parser.parse_args())
    sys.exit(0)