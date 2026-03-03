required:
  Python 3.12
________________________________________
First step: Before running the application, get the packages from the requirements.txt file.

________________________________________
Create&Connect virtual environment
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\requirements.txt 
python .\main.py