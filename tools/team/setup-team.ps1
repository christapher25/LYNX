# One-time setup for the LYNX tab team (one Claude Code session per tab, coordinated through files).
# Run from the main checkout:  powershell -ExecutionPolicy Bypass -File tools\team\setup-team.ps1
# Safe to re-run: existing worktrees, links and board files are left as they are.
$ErrorActionPreference = 'Continue'   # native tools write progress to stderr; exit codes are checked instead
$root = 'C:\Users\chris\PycharmProjects'
$repo = Join-Path $root 'LYNX_V1'
$team = Join-Path $root 'LYNX-team'
$kit  = Join-Path $repo 'tools\team'
$worktrees = @('LYNX_V1-devA', 'LYNX_V1-devB', 'LYNX_V1-test', 'LYNX_V1-verify', 'LYNX_V1-sec')
function Step($t) { Write-Host "`n== $t" -ForegroundColor Cyan }
function Fail($t) { Write-Host "STOP: $t" -ForegroundColor Red; exit 1 }

Step 'Checking tools'
foreach ($c in @('git', 'claude', 'wt')) {
    if (-not (Get-Command $c -ErrorAction SilentlyContinue)) { Fail "'$c' is not installed or not on PATH." }
}
if (-not (Test-Path (Join-Path $repo '.venv\Scripts\python.exe'))) { Fail "No virtual environment at $repo\.venv" }
# cmd.exe tabs may not see the PATH PowerShell sees, so the start files use claude's full path.
$claudeCmd = (Get-Command claude -ErrorAction SilentlyContinue).Source
if ($claudeCmd -and $claudeCmd.ToLower().EndsWith('.ps1')) {
    $alt = [IO.Path]::ChangeExtension($claudeCmd, '.cmd')
    if (Test-Path $alt) { $claudeCmd = $alt } else { Fail "claude resolves to $claudeCmd, which cmd.exe cannot run. Tell Claude." }
}
Write-Host "claude found at: $claudeCmd"
Set-Location $repo
git config core.longpaths true

Step 'Team folder'
foreach ($d in @('', 'tasks', 'reports', 'out', 'roles', 'bin')) { New-Item -ItemType Directory -Force -Path (Join-Path $team $d) | Out-Null }
Copy-Item -Path (Join-Path $kit 'roles\*') -Destination (Join-Path $team 'roles') -Force
foreach ($f in Get-ChildItem -Path (Join-Path $kit 'board')) {
    $t = Join-Path $team $f.Name
    if (-not (Test-Path $t)) { Copy-Item -Path $f.FullName -Destination $t } else { Write-Host "kept existing $($f.Name)" }
}
foreach ($f in Get-ChildItem -Path (Join-Path $kit 'tasks')) {
    $t = Join-Path $team "tasks\$($f.Name)"
    if (-not (Test-Path $t)) { Copy-Item -Path $f.FullName -Destination $t } else { Write-Host "kept existing tasks\$($f.Name)" }
}
# Tab sessions must not form agent teams of their own.
Set-Content -Path (Join-Path $team 'bin\tab-settings.json') -Value '{"env":{"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS":"0"}}' -Encoding ASCII

Step 'Worktrees (code is per worktree; data folders are shared links to the main checkout)'
$linkDirs = @('data\processed', 'data\constituents', 'models\checkpoints', 'models\graph')
foreach ($w in $worktrees) {
    $p = Join-Path $root $w
    if (Test-Path $p) { Write-Host "exists: $p (left as it is)" }
    else {
        git worktree add --detach $p main
        if ($LASTEXITCODE -ne 0) { Fail "git worktree add failed for $p" }
        Write-Host "created: $p"
    }
    # data\raw holds tracked DATA_NOTES.md, so link each data\raw subfolder instead of the folder itself.
    $rawMain = Join-Path $repo 'data\raw'
    New-Item -ItemType Directory -Force -Path (Join-Path $p 'data\raw') | Out-Null
    foreach ($sub in Get-ChildItem -Path $rawMain -Directory) {
        $dst = Join-Path $p "data\raw\$($sub.Name)"
        if (-not (Test-Path $dst)) { New-Item -ItemType Junction -Path $dst -Target $sub.FullName | Out-Null }
    }
    foreach ($rel in $linkDirs) {
        $src = Join-Path $repo $rel; $dst = Join-Path $p $rel
        if ((Test-Path $src) -and -not (Test-Path $dst)) {
            New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
            New-Item -ItemType Junction -Path $dst -Target $src | Out-Null
        }
    }
    $envSrc = Join-Path $repo '.env'; $envDst = Join-Path $p '.env'
    if ((Test-Path $envSrc) -and -not (Test-Path $envDst)) { New-Item -ItemType HardLink -Path $envDst -Target $envSrc | Out-Null }
    # Imports must resolve to THIS worktree's code, not the main checkout's.
    Push-Location $p
    $where = & (Join-Path $repo '.venv\Scripts\python.exe') -c "import vanta, os; print(os.path.dirname(os.path.dirname(vanta.__file__)))"
    Pop-Location
    if ($where -and ($where.Trim().TrimEnd('\') -ne $p)) { Write-Host "  WARNING: in $w, 'import vanta' resolves to $where, not $p. Tell Claude." -ForegroundColor Yellow }
    else { Write-Host "  imports OK in $w" }
}

Step 'Start files, one per tab'
$lead = @{ Name = 'LEAD'; Dir = $repo; Model = ''; Every = '15m'; Role = 'lead'; Extra = '--remote-control' }
$roles = @(
    $lead,
    @{ Name = 'DEV-A';    Dir = (Join-Path $root 'LYNX_V1-devA');   Model = 'sonnet'; Every = '10m'; Role = 'dev';      Extra = '' },
    @{ Name = 'DEV-B';    Dir = (Join-Path $root 'LYNX_V1-devB');   Model = 'sonnet'; Every = '10m'; Role = 'dev';      Extra = '' },
    @{ Name = 'TESTER';   Dir = (Join-Path $root 'LYNX_V1-test');   Model = 'sonnet'; Every = '20m'; Role = 'tester';   Extra = '' },
    @{ Name = 'SECURITY'; Dir = (Join-Path $root 'LYNX_V1-sec');    Model = 'sonnet'; Every = '20m'; Role = 'security'; Extra = '' },
    @{ Name = 'VERIFIER'; Dir = (Join-Path $root 'LYNX_V1-verify'); Model = 'sonnet'; Every = '20m'; Role = 'verifier'; Extra = '' }
)
$lines = @()
foreach ($r in $roles) {
    $loop = "/loop $($r.Every) You are $($r.Name). Follow $team\roles\$($r.Role).md and run one cycle."
    $model = ''; if ($r.Model) { $model = "--model $($r.Model) " }
    $cmd = @(
        '@echo off',
        "title $($r.Name)",
        "cd /d $($r.Dir)",
        "call $repo\.venv\Scripts\activate.bat",
        "`"$claudeCmd`" $model--permission-mode auto --settings $team\bin\tab-settings.json --add-dir $team --name $($r.Name) $($r.Extra) `"$loop`""
    ) -join "`r`n"
    Set-Content -Path (Join-Path $team "bin\start-$($r.Name).cmd") -Value $cmd -Encoding ASCII
    $lines += "$($r.Name): $loop"
}
Set-Content -Path (Join-Path $team 'START-LINES.txt') -Value $lines -Encoding UTF8

Step 'Done'
git worktree list
Write-Host "`nNext: powershell -ExecutionPolicy Bypass -File tools\team\launch-team.ps1" -ForegroundColor Green
