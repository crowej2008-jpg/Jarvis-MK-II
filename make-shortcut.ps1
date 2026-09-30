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

# Resolve the Desktop the way Explorer does. On a OneDrive-backed account the
# registry value is the real Desktop (C:\Users\<you>\OneDrive\Desktop) while
# [Environment]::GetFolderPath can hand back the bare C:\Users\<you>\Desktop,
# which on this machine does not exist at all. A shortcut saved into a path that
# is not there is a shortcut that quietly did not get created, so the registry
# wins and the result is verified before this claims success.
$desktop = (Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders").Desktop
if ([string]::IsNullOrWhiteSpace($desktop)) {
    $desktop = [Environment]::GetFolderPath("Desktop")
}
if (-not (Test-Path -LiteralPath $desktop)) {
    New-Item -ItemType Directory -Path $desktop -Force | Out-Null
}

$lnkPath = Join-Path $desktop "JARVIS - HUD.lnk"

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($lnkPath)
$lnk.TargetPath = $wscript
$lnk.Arguments = "`"$vbs`""
$lnk.WorkingDirectory = $PSScriptRoot
$lnk.Description = "Start the JARVIS heads-up display (microphone on)"
$lnk.Save()

if (-not (Test-Path -LiteralPath $lnkPath)) {
    Write-Error "Saved the shortcut but it is not at $lnkPath; check the Desktop path."
}

Write-Host "Created $lnkPath"