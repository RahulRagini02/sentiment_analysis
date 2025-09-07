1.store all file in nlp_pipeline folder

2.come to my room and collect mymodel file.

3. create a models folder inside of predict label folder.

4.in the models folder store my_model folder , label_encoder.pkl file.

5.then in the vs code terminal-:

   1.python -m venv myenv
   
   2.myenv/scripts/activate
   
   3.pip freeze > requirements.txt 
   
   4.in one terminal run 
       1.cd predict_labels
       2.uvicorn backend:app --reload --port 8000
       
   5.in another terminal run 
       1.cd predict_labels
       2.streamlit run frontend.py
       
   6.in another terminal run 
       1.cd sentiment_analysis
       2.uvicorn backend:app --reload --port 8500

    7.in another terminal run 
      1.cd sentiment_analysis
      2.streamlit run frontend.py

6.then go to
    1.sentiment analysis localhost frontend website it is your working file.

for api_key- contact to Mahim.....
      
