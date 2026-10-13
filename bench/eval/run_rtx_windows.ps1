# Equal-memory head-to-head on a Windows laptop with an NVIDIA GPU (written for an RTX 5060 with 8 GB).
#
# Run from the repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File bench\eval\run_rtx_windows.ps1
#   powershell -ExecutionPolicy Bypass -File bench\eval\run_rtx_windows.ps1 -Gsm 60 -He 40   # quick version
#
# Uses PrismML's official Windows CUDA 13.3 build (no compiler needed), downloads the three models
# (resumable), and runs bench\eval\run_eval.py on each. Results go to bench\eval\results-rtx\.
# Bonsai uses the 1.75-bit PTQ1_0 file: same ternary weights as PQ2_0, and it fits in 8 GB.
# The 7.3 GB Qwen3.8 file does not fit in 8 GB; the server's --fit puts the rest on the CPU.
param(
    [int]$Gsm = 250,
    [int]$He = 164,
    [switch]$DeleteModels
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$root    = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$runtime = Join-Path $root "gpu-runtime"
$models  = Join-Path $root "models"
$out     = Join-Path $root "bench\eval\results-rtx"
$tag     = "prism-b10772-e8fd209"
$base    = "https://github.com/PrismML-Eng/llama.cpp/releases/download/$tag"

# 1. checks
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) { throw "nvidia-smi not found: install the NVIDIA driver first" }
& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# PATH first, then per-user installs (a fresh winget install is not on PATH until a new terminal)
$python = $null
$candidates = @((Get-Command python -ErrorAction SilentlyContinue).Source) +
              @(Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | ForEach-Object FullName)
foreach ($c in $candidates | Where-Object { $_ }) {
    try {
        $v = & $c --version 2>$null
        if ($v -match "Python 3\.(\d+)" -and [int]$Matches[1] -ge 8) { $python = $c; break }
    } catch {}
}
if (-not $python) { throw "Python 3.8+ not found: run  winget install Python.Python.3.12" }
Write-Host "python: $python"

# 2. runtime (official PrismML Windows CUDA build, pinned)
$server = Get-ChildItem $runtime -Recurse -Filter llama-server.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $server) {
    New-Item -ItemType Directory -Force $runtime | Out-Null
    foreach ($zip in @("llama-$tag-bin-win-cuda-13.3-x64.zip", "cudart-llama-bin-win-cuda-13.3-x64.zip")) {
        Write-Host "downloading $zip"
        $dst = Join-Path $runtime $zip
        & curl.exe -L --fail --retry 5 -C - -o $dst "$base/$zip"
        if ($LASTEXITCODE -ne 0) { throw "download failed: $zip" }
        Expand-Archive -Force $dst $runtime
        Remove-Item $dst
    }
    $server = Get-ChildItem $runtime -Recurse -Filter llama-server.exe | Select-Object -First 1
}
# the cudart DLLs must sit next to llama-server.exe
Get-ChildItem $runtime -Recurse -Filter "cu*.dll" | Where-Object { $_.DirectoryName -ne $server.DirectoryName } |
    ForEach-Object { Copy-Item $_.FullName $server.DirectoryName -Force }
Write-Host "server: $($server.FullName)"

# 3. models, resumable
$list = @(
    @{ name = "bonsai2-27b-ptq1_0";  repo = "prism-ml/Ternary-Bonsai-2-27B-gguf"; file = "Ternary-Bonsai-2-27B-PTQ1_0.gguf" },
    @{ name = "qwen3.8-27b-iq2_xxs"; repo = "unsloth/Qwen3.8-27B-GGUF";           file = "Qwen3.8-27B-UD-IQ2_XXS.gguf" },
    @{ name = "qwen3-8b-q6_k";       repo = "Qwen/Qwen3-8B-GGUF";                 file = "Qwen3-8B-Q6_K.gguf" }
)
New-Item -ItemType Directory -Force $models, $out | Out-Null

foreach ($m in $list) {
    if (Test-Path (Join-Path $out "$($m.name).summary.json")) { Write-Host "$($m.name): already done"; continue }
    $path = Join-Path $models $m.file
    $url  = "https://huggingface.co/$($m.repo)/resolve/main/$($m.file)"
    # Hugging Face reports the real file size in x-linked-size; a complete file is not downloaded again
    $head = & curl.exe -sI $url
    $size = [int64](($head | Select-String -Pattern '^x-linked-size:\s*(\d+)' | Select-Object -First 1).Matches.Groups[1].Value)
    for ($try = 1; $try -le 20; $try++) {
        if ((Test-Path $path) -and $size -gt 0 -and (Get-Item $path).Length -eq $size) { break }
        & curl.exe -L --fail --retry 5 -C - -o $path $url
        Write-Host "download stopped (curl exit $LASTEXITCODE), checking size and resuming if needed..."
        Start-Sleep -Seconds 3
    }
    if (-not ((Test-Path $path) -and (Get-Item $path).Length -eq $size)) { throw "could not download $($m.file)" }
    Write-Host "$($m.file): $([math]::Round($size / 1GB, 2)) GB ready"

    Write-Host "=== $($m.name)"
    $argsList = @("bench\eval\run_eval.py", "--model", $path, "--name", $m.name, "--gsm", $Gsm, "--he", $He,
                  "--gpu-auto", "--server", $server.FullName, "--out", $out)
    Push-Location $root
    & $python -u @argsList
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { throw "eval failed for $($m.name), see $out\$($m.name).server.log" }
    if ($DeleteModels) { Remove-Item $path }
}

# 4. table
Write-Host ""
Write-Host "| model | GB | GSM8K | HumanEval | gen tokens per task | decode tok/s |"
Write-Host "|---|---|---|---|---|---|"
foreach ($f in Get-ChildItem $out -Filter *.summary.json) {
    $r = Get-Content $f.FullName -Raw | ConvertFrom-Json
    $g = $r.gsm8k; $h = $r.humaneval
    $per = [math]::Round(($g.gen_tokens + $h.gen_tokens) / [math]::Max(1, $g.n + $h.n))
    Write-Host ("| {0} | {1:N2} | {2}/{3} ({4:N1}%) | {5}/{6} ({7:N1}%) | {8} | {9:N1} |" -f $r.name, $r.model_gb,
        $g.correct, $g.n, 100 * $g.accuracy, $h.correct, $h.n, 100 * $h.accuracy, $per, $g.decode_tok_s)
}
Write-Host ""
Write-Host "results: $out"
