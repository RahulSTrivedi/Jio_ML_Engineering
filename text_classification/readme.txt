run the following commands:

python -m venv venv

venv\Scripts\activate

if scripts running is restrited run this:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

pip install -r requirements.txt

python model/train.py

python -m uvicorn app.main:app --reload

deactivate

