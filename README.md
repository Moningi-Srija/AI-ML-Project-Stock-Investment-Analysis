# AI-ML-Project-Stock-Investment-Analysis
This repo contains the code files of CS 337 Course Project


1. __download_10k.py__: This code gets the list of stocks from wikipedia, uses them to find the link to their corresponding to 10-K filings htmls from financial modelling prep API. The 10-K filings will be obtained from SEC-Edgar Database from SEC Website.
2. __convert_html_to_pdf.py__: This converts all html files to pdfs. PDFs are preferred due to their token efficiency for further analysis.
3. __make_targets.py__: Generates and stores the Dataframe of stocks with target values in a pickle file.
4. __embeddings_save_gpu.py__: Generates embeddings of pdf files and saves them in ChromaDB.
5. __gpt_scores_as_features_old.py__: Generates features by querying all questions in questions.json and creates final dataframes for test and train.
6. __modeling_and_return_estimation.ipynb__: Does Modelling, estimates results and generates plots.
7. __questions.json__: Contains the questions asked to generate features.
8. __csv_to_pkl.py__: converts a csv file to pkl file.
9. __view_pkl.py__: converts pkl file to csv file.
10. __requirements.txt__: Contains the info about all the packages with versions used in the project.

The order of running files are download_10k.py -> convert_html_to_pdf.py -> make_targets.py -> embeddings_save_gpu.py -> gpt_scores_as_features_old.py -> modelling_and_return_estimation.ipynb.
The paths to folders should be changed to the local system file paths while running following the comments in the files.
