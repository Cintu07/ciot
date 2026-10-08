# Chat with Ternary Bonsai 2 27B on this laptop's CPU, using the CIOT kernels.
# Usage (PowerShell, from this folder):   .\run_bonsai.ps1            # 6 threads, cooler
#                                         .\run_bonsai.ps1 -Threads 8 # fastest, hotter
# Opens http://127.0.0.1:8080 in your browser. Press Ctrl+C in this window to stop the model.
param(
    [int]$Threads = 6,
    [int]$Context = 8192,
    [int]$Port = 8080
)

$root   = $PSScriptRoot
$server = Join-Path $root "vendor\prism-llama.cpp\build\bin\llama-server.exe"
$model  = Join-Path $root "models\Ternary-Bonsai-2-27B-PQ2_0.gguf"

if (-not (Test-Path $server)) { Write-Error "llama-server not built: $server"; exit 1 }
if (-not (Test-Path $model))  { Write-Error "model missing: $model"; exit 1 }

$env:CIOT_T2 = "1"
Write-Host "Loading Bonsai 2 27B (about 7 GB into RAM, takes ~30-60 s)..."
Start-Job -ScriptBlock {
    param($p)
    for ($i = 0; $i -lt 120; $i++) {
        try { if ((Invoke-RestMethod "http://127.0.0.1:$p/health" -TimeoutSec 2).status -eq "ok") { Start-Process "http://127.0.0.1:$p"; return } } catch {}
        Start-Sleep -Seconds 2
    }
} -ArgumentList $Port | Out-Null

& $server -m $model -t $Threads -tb $Threads --load-mode none -c $Context --port $Port --host 127.0.0.1
