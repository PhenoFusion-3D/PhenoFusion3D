<#
launch.ps1 -- PhenoFusion3D Windows launcher.

Activates the Windows venv at .\venv and runs the GUI. Use this instead of
launching from inside WSL: an Intel RealSense camera is a USB device on the
Windows host and is not visible to a Python interpreter running under WSL2
without explicit usbipd-win passthrough (which is fragile and not how this
project is intended to run).

Usage (PowerShell, from any directory):
    pwsh -File path\to\launch.ps1
or simply double-click launch.ps1 if PowerShell is your default for .ps1.
#>

$ErrorActionPreference = "Stop"

# Resolve to repo root regardless of where the script was invoked from.
Set-Location -Path $PSScriptRoot

if (-not (Test-Path .\venv\Scripts\Activate.ps1)) {
    Write-Error @"
.\venv\Scripts\Activate.ps1 not found.

Set up the Windows venv first:
    py -3.11 -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -e ".[windows,l515]"      # for L515 owners
    pip install -e ".[windows]"           # for D400 / D500 owners

Then re-run launch.ps1.
"@
}

. .\venv\Scripts\Activate.ps1

# Sanity hint: if someone somehow runs this from a Bash-flavoured shell on
# Windows, $env:VIRTUAL_ENV won't be set even after dot-sourcing. Bail loudly.
if (-not $env:VIRTUAL_ENV) {
    Write-Error "venv activation did not take effect. Run this from PowerShell, not Git Bash / WSL."
}

Write-Host "[launch] active venv: $env:VIRTUAL_ENV"

# Belt-and-braces: confirm the venv actually has the critical deps installed
# before handing control to main.py. Catches the case where a previous
# `pip install` was interrupted (Ctrl-C, terminal closed, IDE backgrounded the
# command, ...) and main.py would otherwise crash with
#     ModuleNotFoundError: No module named 'PyQt5.QtWidgets'
# from line 23 *before* the startup self-check could run.
#
# Capture via temp file rather than `2>&1` because $ErrorActionPreference =
# "Stop" (and PS 7.x's $PSNativeCommandUseErrorActionPreference) treat any
# native-command stderr as a terminating error, which would interrupt this
# script before our friendly Write-Error message can run.
$probeScript = New-TemporaryFile
Rename-Item $probeScript.FullName ($probeScript.FullName + '.py')
$probeScriptPath = $probeScript.FullName + '.py'
Set-Content -Path $probeScriptPath -Value 'import PyQt5.QtWidgets, pyrealsense2' -Encoding ASCII

$probeOutLog = New-TemporaryFile
$probeErrLog = New-TemporaryFile
try {
    $probeProc = Start-Process -FilePath python `
        -ArgumentList $probeScriptPath `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $probeOutLog.FullName `
        -RedirectStandardError  $probeErrLog.FullName
    if ($probeProc.ExitCode -ne 0) {
        $probeOut = (((Get-Content $probeOutLog.FullName -Raw) + "`n" + (Get-Content $probeErrLog.FullName -Raw))).Trim()
        Write-Error @"
The venv at $env:VIRTUAL_ENV is missing required packages.

Underlying error:
$probeOut

Re-run the install from this PowerShell window:
    .\venv\Scripts\Activate.ps1
    pip install -e ".[windows,l515]"      # for L515 owners
    pip install -e ".[windows]"           # for D400 / D500 owners

Then re-run launch.bat (or launch.ps1).
"@
    }
} finally {
    Remove-Item $probeScriptPath, $probeOutLog.FullName, $probeErrLog.FullName -ErrorAction SilentlyContinue
}

python main.py
