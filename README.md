## Summarization App build by streamlit
#### Using Langchain library to load, split pdf file and then applying model from Huggingface: https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M to summarize the pdf file.  
#### Don't forget to put your pdf file which you want to summarize in Data folder.
```python
# create python virtual environment
python -m venv .venv

# activate env (Window)
.venv/Scripts/activate.ps1 ## powershell
.venv/Scripts/activate.bat

# install all the packages
pip install -r requirements.txt

# run streamlit app
streamlit run app.py
```

