# LYNX nightly health check: runs Claude Code headless (claude -p) with a read-only prompt.
# Scheduled by Windows Task Scheduler (see the setup guide, Step 11). Writes logs\nightly\.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# Use the project's virtual environment so `python` means the venv interpreter.
& (Join-Path $repo ".venv\Scripts\Activate.ps1")

$logDir = Join-Path $repo "logs\nightly"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$out = Join-Path $logDir "run_$ts.json"
$err = Join-Path $logDir "run_$ts.err.log"

# Pass the prompt as one line: if `claude` resolves to a .cmd shim, a line break would cut it short.
$prompt = (Get-Content (Join-Path $repo "docs\agent-team\NIGHTLY_PROMPT.md") -Raw) -replace "\s*\r?\n\s*", " "
$prompt = $prompt.Trim()

# Windows PowerShell 5.1 turns a native command's stderr lines into errors, and with "Stop" the
# first warning would end this script. Let the run finish and keep stderr in its own file.
$ErrorActionPreference = "Continue"
# auto mode + no permission prompts: anything that would need a person is denied, not waited on.
claude -p $prompt --permission-mode auto --permission-prompts none --output-format json > $out 2> $err
exit $LASTEXITCODE
