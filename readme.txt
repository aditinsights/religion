# how to run this program 
    # set up a virtual environment (python -m venv env)
    # install the requirements.txt file (pip install -r requirements.txt) 
    # create a .env file with OPENAI_API_KEY
    # run the app.py file (python -m app)
    # go to the local server and upload a document 
    # select the search terms 
    # run the analysis 
    # view the results

# utils.py 
    # all core functions here 
    # can change the underlying LLM here - right now set to gpt-turbo 
    # can make decisions on the sensitivity of the vector search here 
    # can make decisions on the number of vectors to search here + can set up a true vector db

# embed_generation.py
    # this function creates embeddings from the document uploaded - costs money 
    # does not need to be run if text analysis selected 

# embed_analysis.py 
    # approach to identifying bias by comparing the embeddings of the document to the embeddings of the search terms

# text_analysis.py 
    # approach to identifying bias by comparing the text of the document to the search terms

#app.py 
    # main file for running the app 
    # can set up the routes here 
    # can set up the html here 
    # can set up the functions that run the analysis here


# Ideas to "Build Your Own" Bias Detector 
    # Three steps: 
        # Gather data + annotate it - types of bias, etc. This becomes your "gold" standard. ~ 500 instances per class of bias is a good target. 
        # Train a model to detect bias - classification model trying multiple techniques such as log reg, XGBoost, or a neural network
        # Check stats - precision, recall, F1 score, to avoid overfitting and other errors in data. 

        