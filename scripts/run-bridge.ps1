$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path (Join-Path $scriptDir "..")
$bridgeDir = Join-Path $root "vendor\mirofish-oauthbridge\codex-bridge"

if (-not (Test-Path $bridgeDir)) {
    Write-Host "codex-bridge not found: $bridgeDir"
    exit 1
}

Push-Location $bridgeDir
try {
    if (-not (Test-Path (Join-Path $bridgeDir "node_modules"))) {
        Write-Host "Installing bridge dependencies..."
        npm install
    }

    $env:PORT = "8787"
    $env:BRIDGE_PROVIDER = "gemini"
    $env:GEMINI_MODEL = "gemini-2.5-flash"
    $env:CODEX_BRIDGE_WORKDIR = $root.Path

    Write-Host "Starting codex-bridge on http://127.0.0.1:8787 ..."
    npm start
}
finally {
    Pop-Location
}
