@echo off
REM # comment line # To run this file at startup user windows key +R >> Run Box "shell:startup" put this file or copy one this file and restart the Computer
CALL C:\Users\Office\anaconda3\Scripts\activate.bat C:\Users\Office\anaconda3\envs\pytorch


CD C:\Users\Office\Downloads\test
python backend_server.py