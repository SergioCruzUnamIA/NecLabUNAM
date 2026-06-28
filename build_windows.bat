@echo off
REM Build NecLab.exe for Windows
REM Requirements: pip install pyinstaller  (Python 3.9+, on Windows)

echo =^> Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

echo =^> Running PyInstaller...
pyinstaller NecLab.spec

echo.
echo =^> Done! Executable is at: dist\NecLab.exe
pause
