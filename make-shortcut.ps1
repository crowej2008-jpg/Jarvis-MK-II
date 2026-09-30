# Creates the "JARVIS" Desktop shortcut that starts the HUD.
#
# The shortcut points at wscript.exe running Start-Jarvis-HUD.vbs, so the
# shortcut itself never opens a console and the only visible thing is the JARVIS
# window. Re-running this file updates an existing shortcut rather than
# stranding a second one next to it.

$ErrorActionPreference = "Stop"

$vbs = Join-Path $PSScriptRoot "Start-Jarvis-HUD.vbs"
if (-not (Test-Path -LiteralPath $vbs)) {
    Write-Error "Start-Jarvis-HUD.vbs not found next to this script ($vbs)"
}

$wscript = "$env:WINDIR\System32\wscript.exe"
if (-not (Test-Path -LiteralPath $wscript)) {
    Write-Error "wscript.exe not found"
}

$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "JARVIS - HUD.lnk"

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($lnkPath)
$lnk.TargetPath = $wscript
$lnk.Arguments = "`"$vbs`""
$lnk.WorkingDirectory = $PSScriptRoot
$lnk.Description = "Start the JARVIS heads-up display (microphone on)"
$lnk.Save()

Write-Host "Created $lnkPath"