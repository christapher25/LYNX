Set-Location (Split-Path -Parent $PSScriptRoot)
& .\.venv\Scripts\Activate.ps1
$env:CLAUDE_CODE_ENABLE_TODO_TOOLS = "1"
claude --model opus --permission-mode auto --name "LYNX lead" --remote-control
