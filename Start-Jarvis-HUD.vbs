' Launch the JARVIS heads-up display with no console window behind it.
'
' WHY THIS IS A .VBS AND NOT A .BAT
' ---------------------------------
' The display is a GUI. Under pythonw.exe there is no console, which is the
' point, but it also means a traceback goes nowhere at all: a bad config, a
' missing tkinter, an Ollama that never started would all look identical to
' the user, namely a window that never appears. So this launcher runs the app
' with its output redirected to a log, and if the process exits with a failure
' it reads the tail of that log and puts it in a message box. A launcher that
' fails quietly is worse than no launcher.
'
' Usage: Start-Jarvis-HUD.vbs          normal, keyboard and microphone
'        Start-Jarvis-HUD.vbs text     keyboard only

Option Explicit

Dim fso, shell, base, logPath, logDir, pythonw, cmd, rc, tail

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

base = fso.GetParentFolderName(WScript.ScriptFullName)
logDir = shell.ExpandEnvironmentStrings("%USERPROFILE%\.jarvis")
If Not fso.FolderExists(logDir) Then
  fso.CreateFolder logDir
End If
logPath = logDir & "\hud-launch.log"

pythonw = FindPythonw()
If pythonw = "" Then
  MsgBox "Could not find pythonw.exe." & vbCrLf & vbCrLf & _
         "JARVIS needs Python installed. Try running:" & vbCrLf & _
         "    python -m jarvis --hud --voice", vbCritical, "JARVIS"
  WScript.Quit 1
End If

' --hud for the display, --voice to open the microphone in it. Extra arguments
' are passed through, so "text" above runs it keyboard only.
Dim args, i, extra
args = ""
If WScript.Arguments.Count > 0 Then
  For i = 0 To WScript.Arguments.Count - 1
    args = args & " " & WScript.Arguments(i)
  Next
End If

' One JARVIS at a time. Two instances both open the microphone and the ack
' stream, and the second one wins the device while the first keeps displaying a
' mic that is no longer its own. Refusing is also the honest answer to somebody
' who clicks again because the window has not appeared yet.
If JarvisAlreadyRunning() Then
  MsgBox "JARVIS is already running." & vbCrLf & vbCrLf & _
         "Look for the J.A.R.V.I.S. window, or check the taskbar. Starting a " & _
         "second copy would take the microphone away from the first one.", _
         vbExclamation, "JARVIS"
  WScript.Quit 0
End If

' Start marker so the log can be trimmed to just this run's output; without it
' the tail below could quote an error from an earlier session.
'
' This must never be able to stop the launcher. A second launch happens the
' moment the first one looks like it did nothing - the window is slow, or it is
' behind something - and by then the first instance's cmd redirection is still
' holding this log open, so appending to it fails with "Permission denied"
' (800A0046). That is cosmetic information failing loudly and killing the app
' launch with it, so it is swallowed and the marker is simply skipped.
Dim log, marker
marker = "=== run " & Now & " ==="
On Error Resume Next
Set log = fso.OpenTextFile(logPath, 8, True)
If Err.Number = 0 Then
  log.WriteLine marker
  log.Close
End If
Err.Clear
On Error GoTo 0

' Window style 0 is "hidden". Going through cmd is what makes the redirection
' possible; pythonw cannot redirect its own output on Windows.
cmd = "cmd /c ""cd /d """ & base & """ && """ & pythonw & """ -m jarvis --hud --voice" & args & " >> """ & logPath & """ 2>&1"""

rc = shell.Run(cmd, 0, True)

If rc <> 0 Then
  tail = ReadTail(logPath, 1400)
  MsgBox "JARVIS stopped with exit code " & rc & "." & vbCrLf & vbCrLf & _
         "Log tail:" & vbCrLf & tail & vbCrLf & vbCrLf & _
         "Full log: " & logPath, vbCritical, "JARVIS"
End If

WScript.Quit rc

' Newest Python first. Folder names like "Python313" and "Python39" do not sort
' correctly as strings, so the version digit is parsed and compared numerically.
Function FindPythonw()
  Dim root, subf, cand, candidate, best, digits, rev
  FindPythonw = ""
  rev = 0
  best = ""
  root = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%\Programs\Python")
  If Not fso.FolderExists(root) Then
    root = ""
  End If
  Do While True
    Dim folder
    folder = root
    If root = "" Then
      ' Nothing under %LOCALAPPDATA%; try a PATH lookup once instead.
      Set subf = shell.Exec("cmd /c where pythonw.exe 2>nul")
      If subf Is Nothing Or Err.Number <> 0 Then Exit Do
      Dim out
      out = Trim(subf.StdOut.ReadAll)
      If out = "" Then Exit Do
      FindPythonw = Split(out, vbCrLf)(0)
      Exit Do
    End If
    For Each cand In fso.GetFolder(folder).SubFolders
      candidate = cand.Path & "\pythonw.exe"
      If fso.FileExists(candidate) Then
        digits = cand.Name
        Dim i
        i = InStrRev(cand.Name, "Python")
        If i > 0 Then digits = Mid(cand.Name, i + 6)
        digits = Replace(digits, "-", ".")
        If IsNumeric(digits) And CDbl(digits) > rev Then
          rev = CDbl(digits)
          best = candidate
        End If
      End If
    Next
    Exit Do
  Loop
  If best <> "" Then FindPythonw = best
End Function

Function JarvisAlreadyRunning()
  ' Any live pythonw running "-m jarvis" is a session that already owns the
  ' microphone. WMI can fail on a locked-down box, and then this returns False,
  ' which is the old behaviour rather than refusing to start. Iterating the
  ' collection rather than testing .Count avoids the -1 that a forward-only
  ' result set reports.
  On Error Resume Next
  JarvisAlreadyRunning = False
  Dim svc, coll, hit
  Set svc = GetObject("winmgmts:\\.\root\cimv2")
  If Err.Number <> 0 Then Exit Function
  Set coll = svc.ExecQuery( _
    "SELECT ProcessId FROM Win32_Process " & _
    "WHERE Name='pythonw.exe' AND CommandLine LIKE '%-m jarvis%'")
  If Err.Number <> 0 Then Exit Function
  For Each hit In coll
    JarvisAlreadyRunning = True
    Exit Function
  Next
End Function

Function ReadTail(path, maxChars)
  Dim text
  If Not fso.FileExists(path) Then
    ReadTail = "(no log was written)"
    Exit Function
  End If
  Dim f
  Set f = fso.OpenTextFile(path, 1)
  text = f.ReadAll
  f.Close
  If Len(text) > maxChars Then
    text = "..." & Right(text, maxChars)
  End If
  ReadTail = text
End Function
