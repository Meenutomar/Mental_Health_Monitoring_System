
## How to setup (for the first time) - if virtual environment 'backend-venv' does not exists
1. Open Terminal and go to backend folder
2. Run following command to create virtual environment (if doesn't exists)
<code>
python -m venv backend-venv
</code>

## How to setup (next time onwards)
1. Open Terminal and go to backend folder
2. Fire the following command to activate virtual environment
<code>
backend-venv/Scripts/activate
</code>
This will activate virtual environment

## How to run
<code>
uvicorn main:app
</code>

