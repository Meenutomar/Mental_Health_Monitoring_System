
## How to setup (for the first time) - if virtual environment 'frontend-venv' does not exists
1. Open Terminal and go to frontend folder
2. Run following command to create virtual environment (if doesn't exists)
<code>
python -m venv frontend-venv
</code>

## How to setup (next time onwards)
1. Open Terminal and go to frontend folder
2. Fire the following command to activate virtual environment
<code>
frontend-venv/Scripts/activate
</code>
This will activate virtual environment

## How to start app
Run the following command in terminal
<code>
 streamlit run main.py
</code>

For automatic reload use 
<code>
streamlit run main.py --server.runOnSave true
</code>
