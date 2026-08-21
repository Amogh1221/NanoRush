@echo off
echo Starting NanoRush API Server...
echo Make sure you have activated your virtual environment!
echo If you get an error, please run: pip install -r requirements.txt
echo.
python -m uvicorn main:app --reload
pause
