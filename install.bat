@echo off
setlocal

echo.
echo ========================================
echo    SegmentME Installer (Windows)
echo ========================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found. Install Python 3.10+ from https://python.org and try again.
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo [2/5] Installing PyTorch (CPU)...
pip install --quiet --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo [3/5] Installing dependencies...
pip install --quiet --no-cache-dir -r requirements.txt

rem SAM2, SAM3 and DEXTR are not on PyPI; installed from GitHub archives (no git needed)
echo [4/5] Installing SAM2, SAM3 and DEXTR...
set SAM2_BUILD_CUDA=0
pip install --quiet --no-cache-dir "https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip"
pip install --quiet --no-cache-dir "https://github.com/facebookresearch/sam3/archive/refs/heads/main.zip" einops pycocotools
set DEXTR_TMP=%TEMP%\dextr_sme
rmdir /s /q "%DEXTR_TMP%" >nul 2>&1
mkdir "%DEXTR_TMP%"
curl -sL "https://github.com/StevetheGreek97/DEXTR-SMe/archive/refs/heads/master.zip" -o "%DEXTR_TMP%\dextr.zip"
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "%DEXTR_TMP%\dextr.zip" "%DEXTR_TMP%"
type nul > "%DEXTR_TMP%\DEXTR-SMe-master\src\DEXTR\__init__.py"
pip install --quiet --no-cache-dir "%DEXTR_TMP%\DEXTR-SMe-master"
rmdir /s /q "%DEXTR_TMP%"

rem Model checkpoints live in models\ next to the app. Only SAM2 Tiny is fetched
rem here; everything else can be downloaded from Settings -> Models inside the app.
echo [5/5] Downloading the default SAM2 Tiny model...
if not exist "models" mkdir models

if not exist "models\sam2_hiera_tiny.pt" (
    echo   Downloading sam2_hiera_tiny.pt ~155 MB...
    curl -L --progress-bar "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt" -o "models\sam2_hiera_tiny.pt"
) else (
    echo   sam2_hiera_tiny.pt already present, skipping.
)

rem SAM3 checkpoint (gated on Hugging Face -- needs an account with access)
if not exist "models\sam3.pt" (
    echo   Trying to download sam3.pt ~3.4 GB, gated...
    .venv\Scripts\hf.exe download facebook/sam3 sam3.pt --local-dir models >nul 2>&1
    if errorlevel 1 (
        echo   SAM3 not downloaded (needs Hugging Face access). To add it later:
        echo     request access at https://huggingface.co/facebook/sam3, then run
        echo     .venv\Scripts\hf.exe auth login
        echo     .venv\Scripts\hf.exe download facebook/sam3 sam3.pt --local-dir models
        echo     or import the file from Settings -^> Models inside the app.
    ) else (
        echo   sam3.pt downloaded.
    )
) else (
    echo   sam3.pt already present, skipping.
)

echo.
echo ========================================
echo    Installation complete!
echo    Run run.bat to start SegmentME.
echo    Other models (SAM2 Small/Large, SAM2.1, DEXTR):
echo    Settings -^> Models inside the app.
echo ========================================
echo.
pause
