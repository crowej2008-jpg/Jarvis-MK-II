@echo off
rem Debug launcher for the JARVIS heads-up display.
rem This one keeps a console open so tracebacks stay visible; the day-to-day
rem launcher is Start-Jarvis-HUD.vbs, which runs pythonw.exe and reports
rem failures in a message box instead.
cd /d "%~dp0"
python -m jarvis --hud --voice
echo.
echo JARVIS closed (exit code %ERRORLEVEL%).
pause