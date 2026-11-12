# Opens one Windows Terminal window with a tab per role.
# Usage: powershell -ExecutionPolicy Bypass -File tools\team\launch-team.ps1 [-Roles LEAD,DEV-A,TESTER]
param([string[]]$Roles = @('LEAD', 'DEV-A', 'DEV-B', 'TESTER', 'SECURITY', 'VERIFIER'))
$bin = 'C:\Users\chris\PycharmProjects\LYNX-team\bin'
$parts = @()
foreach ($r in $Roles) {
    $f = Join-Path $bin "start-$r.cmd"
    if (-not (Test-Path $f)) { Write-Host "Missing $f. Run tools\team\setup-team.ps1 first." -ForegroundColor Red; exit 1 }
    $parts += "new-tab --title $r cmd /k `"$f`""
}
$wtArgs = '-w lynx ' + ($parts -join ' ; ')
Start-Process -FilePath 'wt.exe' -ArgumentList $wtArgs
Write-Host "Opened: $($Roles -join ', ')"
Write-Host 'If a tab shows the /loop line as a plain message instead of starting a loop, paste that tab''s line from C:\Users\chris\PycharmProjects\LYNX-team\START-LINES.txt.'
