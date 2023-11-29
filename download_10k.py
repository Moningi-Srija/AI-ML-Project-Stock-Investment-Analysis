import requests
import json
import pandas as pd
import os
import argparse
import sys

def download_report(url, path):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    file_extension = url.split('.')[-1]
    path = path + '.' + file_extension
    if response.status_code == 200:
        # Get the content of the file
        page_content = response.content

        # Write the PDF content to the local file
        with open(path, "wb") as file:
            file.write(page_content)
    else:
        raise ValueError('Response not 200. Broken for: {}'.format(url))

def get_all_tickers():
    '''
    Function to fetch the list of stocks in various US market indices
    '''
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    ticker_list_500 = sp500[0].Symbol.to_list()
    # sp400 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')
    # ticker_list_400 = sp400[0].Symbol.to_list()
    # sp600 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')
    # ticker_list_600 = sp600[0].Symbol.to_list()
    # ticker_list = list(set(ticker_list_500 + ticker_list_400 + ticker_list_600))
    ticker_list = sorted(list(set(ticker_list_500)))
    return ticker_list

def main(args):
    # with open(args.config_path) as json_file:
    #     config_dict = json.load(json_file)
    ticker_list = get_all_tickers()
    i = 0
    for ticker in ticker_list[5:]:
        print(ticker)
        check_saved_path = os.path.join('https://financialmodelingprep.com/api/v3/sec_filings/{}?type=10-K&page=0&apikey={}', ticker)
        if os.path.exists(check_saved_path):
            continue
        fmp_10k_url = 'https://financialmodelingprep.com/api/v3/sec_filings/{}?type=10-K&page=0&apikey={}'.format(ticker,
                                                                                                                  "41909326db0bb0740658aa1c86597b32")
        response = requests.get(fmp_10k_url)
        for d in json.loads(response.content):
            filing_type = d['type']
            if not ((filing_type.lower() == '10-k') | (filing_type.lower() == '10k')):
                continue
            date_string = d['fillingDate']
            date = date_string[:10]
            year = date_string[:4]
            if int(year) < 2002:
                continue
            link = d['finalLink']
            save_path_directory = os.path.join("/media/moningi-srija/Seagate Backup Plus Drive/ai_ml/GPT-InvestAR-main/data/html", ticker, date)
            if not os.path.exists(save_path_directory):
                os.makedirs(save_path_directory)
            save_path = os.path.join(save_path_directory, date)
            download_report(link, save_path)
        i = i+1
        print('Completed: {}/{}'.format(i, len(ticker_list)))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_path', dest='config_path', type=str,
    #                     required=True,
    #                     help='''Full path of config.json''')
    main(args=parser.parse_args())
    sys.exit(0)