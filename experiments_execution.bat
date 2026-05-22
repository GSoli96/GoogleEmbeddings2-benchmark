@echo off
echo ========================================
echo Avvio dei benchmark
echo ========================================

echo.
echo [1/9] Esecuzione: run-italian --models labse
python src/benchmark.py run-italian --models labse
if errorlevel 1 (
    echo Errore durante l'esecuzione del benchmark labse su italiano
    pause
    exit /b 1
)

echo.
echo [2/9] Esecuzione: run-italian --models mpnet
python src/benchmark.py run-italian --models mpnet
if errorlevel 1 (
    echo Errore durante l'esecuzione del benchmark mpnet su italiano
    pause
    exit /b 1
)

echo.
echo [3/9] Esecuzione: run-chunking --models me5
python src/benchmark.py run-chunking --models me5
if errorlevel 1 (
    echo Errore durante l'esecuzione del benchmark me5 su chunking
    pause
    exit /b 1
)

echo.
echo [4/9] Esecuzione: run-chunking --models labse
python src/benchmark.py run-chunking --models labse
if errorlevel 1 (
    echo Errore durante l'esecuzione del benchmark labse su chunking
    pause
    exit /b 1
)

echo.
echo [5/9] Esecuzione: run-chunking --models mpnet
python src/benchmark.py run-chunking --models mpnet
if errorlevel 1 (
    echo Errore durante l'esecuzione del benchmark mpnet su chunking
    pause
    exit /b 1
)

echo.
echo [6/9] Esecuzione: run-beir --models me5
python src/benchmark.py run-beir --models me5
if errorlevel 1 (
    echo Errore durante l'esecuzione del benchmark me5 su beir
    pause
    exit /b 1
)

echo.
echo [7/9] Esecuzione: run-beir --models labse
python src/benchmark.py run-beir --models labse
if errorlevel 1 (
    echo Errore durante l'esecuzione del benchmark labse su beir
    pause
    exit /b 1
)

echo.
echo [8/9] Esecuzione: run-beir --models mpnet
python src/benchmark.py run-beir --models mpnet
if errorlevel 1 (
    echo Errore durante l'esecuzione del benchmark mpnet su beir
    pause
    exit /b 1
)

echo.
echo ========================================
echo Tutti i benchmark completati con successo!
echo ========================================
pause