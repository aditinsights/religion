# how to run this program 
    # set up a virtual environment (python -m venv env)
    # install the requirements.txt file (pip install -r requirements.txt) 
    # create a .env file with OPENAI_API_KEY
    # run the app.py file (python -m app.py)
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


