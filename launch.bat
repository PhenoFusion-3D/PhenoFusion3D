@echo off
REM launch.bat -- PhenoFusion3D Windows launcher (double-click friendly).
REM
REM Wraps launch.ps1 so users don't have to fight Windows PowerShell's
REM script execution policy ("not digitally signed" / UnauthorizedAccess).
REM This .bat is not subject to that policy; it shells out to PowerShell
REM with -ExecutionPolicy Bypass for this one invocation only and does not
REM change any persistent setting on the machine.

setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" %*
endlocal
