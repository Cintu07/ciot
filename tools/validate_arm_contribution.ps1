# Run on the Snapdragon laptop from the CIOT root. No upstream submissions are made.
param(
    [ValidateSet('Prepare','Build','Small','Large')][string]$Stage = 'Small',
    [string]$CompilerC = 'clang',
    [string]$CompilerCxx = 'clang++',
    [string]$Python = 'python',
    [ValidateRange(1,8)][int]$Threads = 4,
    [ValidateRange(1,4)][int]$BuildJobs = 2,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$revision = 'e8fd209c975c80aa7fb36a8423571ff3b1b7e955'
$patch = Join-Path $root 'patches/pr-arm-pq2_0-repack.patch'
$run = Join-Path $root ('bench/contribution/arm-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$sources = @{
    stock = Join-Path $root 'vendor/prism-arm-stock'
    candidate = Join-Path $root 'vendor/prism-arm-candidate'
}
if ($PlanOnly) {
    Write-Host "Revision: $revision; stage: $Stage; CPU threads: $Threads; build jobs: $BuildJobs"
    Write-Host 'Prepare: isolated stock/candidate checkouts, apply the existing patch to candidate.'
    Write-Host 'Build: CPU-only Ninja builds of both versions; no CUDA.'
    Write-Host 'Small: CPU reference tests, paired perplexity/greedy checks, interleaved benchmarks on Qwen3-0.6B PQ2_0.'
    Write-Host 'Large: the same paired model checks on Bonsai 27B PQ2_0; small validation should pass first.'
    Write-Host 'Outputs: timestamped bench/contribution directory. No commit, push, issue, or PR.'
    exit 0
}
if ($Stage -ne 'Prepare') {
    $arch = (Get-CimInstance Win32_Processor | Select-Object -First 1).Architecture
    if ($arch -ne 12) { throw 'This execution stage needs a native ARM64 Windows machine. Use -PlanOnly on x64.' }
    foreach ($compiler in @($CompilerC, $CompilerCxx)) {
        $version = & $compiler --version
        if ($LASTEXITCODE -ne 0 -or ($version -join "`n") -notmatch 'Target:\s*(aarch64|arm64)') {
            throw "Compiler must target ARM64: $compiler"
        }
    }
    if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { throw 'Ninja must be on PATH.' }
}
New-Item -ItemType Directory -Force $run | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Invoke-Recorded([string]$Name, [string]$Exe, [string[]]$NativeArgs) {
    Write-Host "[$Name] $Exe $($NativeArgs -join ' ')"
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $output = & $Exe @NativeArgs 2>&1; $code = $LASTEXITCODE }
    finally { $ErrorActionPreference = $savedPreference }
    $text = ($output | Out-String -Width 4096)
    [IO.File]::WriteAllText((Join-Path $run "$Name.txt"), $text, $utf8)
    [IO.File]::WriteAllText((Join-Path $run "$Name.status.json"), (@{ executable=$Exe; arguments=$NativeArgs; exit_code=$code } | ConvertTo-Json -Depth 5), $utf8)
    if ($code -ne 0) { throw "$Name failed ($code); see $run" }
    return $text
}
foreach ($mode in @('stock','candidate')) {
    $source = $sources[$mode]
    if (-not (Test-Path $source)) {
        Invoke-Recorded "$mode-clone" 'git' @('clone','--no-checkout','--depth','1','--branch','prism','https://github.com/PrismML-Eng/llama.cpp.git',$source) | Out-Null
        Invoke-Recorded "$mode-fetch" 'git' @('-C',$source,'fetch','--depth','1','origin',$revision) | Out-Null
        Invoke-Recorded "$mode-checkout" 'git' @('-C',$source,'checkout','--detach',$revision) | Out-Null
    }
    $head = & git -C $source rev-parse HEAD
    if ($head -ne $revision) { throw "Unexpected revision in $source; preserve it and use another directory." }
    $dirty = & git -C $source status --porcelain
    if ($mode -eq 'stock' -and $dirty) { throw "Stock source is modified: $source" }
    if ($mode -eq 'candidate') {
        if (-not $dirty) {
            Invoke-Recorded 'candidate-apply-check' 'git' @('-C',$source,'apply','--check',$patch) | Out-Null
            Invoke-Recorded 'candidate-apply' 'git' @('-C',$source,'apply',$patch) | Out-Null
        }
        $expected = & git -C $source diff --numstat
        if (@($dirty).Count -gt 1 -or $expected -notmatch 'ggml/src/ggml-cpu/repack.cpp$') { throw 'Candidate contains unrelated changes.' }
        Invoke-Recorded 'candidate-reverse-check' 'git' @('-C',$source,'apply','--reverse','--check',$patch) | Out-Null
        Invoke-Recorded 'candidate-whitespace' 'git' @('-C',$source,'diff','--check') | Out-Null
    }
}
$manifest = @{ revision=$revision; patch_sha256=(Get-FileHash $patch -Algorithm SHA256).Hash; stage=$Stage;
    threads=$Threads; build_jobs=$BuildJobs; processor=(Get-CimInstance Win32_Processor | Select-Object Name,Architecture);
    battery=(Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining) }
[IO.File]::WriteAllText((Join-Path $run 'environment.json'), ($manifest | ConvertTo-Json -Depth 5), $utf8)
if ($Stage -eq 'Prepare') { Write-Host "Prepared: $run"; exit 0 }
$bins = @{}
foreach ($mode in @('stock','candidate')) {
    $build = Join-Path $root "build/prism-arm-$mode"
    Invoke-Recorded "$mode-configure" 'cmake' @('-S',$sources[$mode],'-B',$build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
        "-DCMAKE_C_COMPILER=$CompilerC","-DCMAKE_CXX_COMPILER=$CompilerCxx",'-DGGML_NATIVE=ON','-DGGML_OPENMP=OFF',
        '-DGGML_CUDA=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_SERVER=OFF','-DLLAMA_BUILD_TESTS=ON',
        '-DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00','-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00') | Out-Null
    Invoke-Recorded "$mode-build" 'cmake' @('--build',$build,'--target','test-backend-ops','llama-bench','llama-perplexity','--parallel',"$BuildJobs") | Out-Null
    $bins[$mode] = Join-Path $build 'bin'
}
if ($Stage -eq 'Build') { Write-Host "Built: $run"; exit 0 }
$probeBins = @{}
foreach ($mode in @('stock','candidate')) {
    $probeBuild = Join-Path $root "build/arm-check-$mode"
    Invoke-Recorded "$mode-probe-configure" 'cmake' @('-S',(Join-Path $root 'bench/arm_repack_check'),'-B',$probeBuild,'-G','Ninja',
        '-DCMAKE_BUILD_TYPE=Release',"-DCMAKE_CXX_COMPILER=$CompilerCxx","-DPRISM_SOURCE=$($sources[$mode])",
        "-DPRISM_BUILD=$(Join-Path $root "build/prism-arm-$mode")") | Out-Null
    Invoke-Recorded "$mode-probe-build" 'cmake' @('--build',$probeBuild,'--parallel',"$BuildJobs") | Out-Null
    $probeBins[$mode] = $probeBuild
}
$savedPath = $env:PATH
try {
    $env:PATH = $bins.candidate + ';' + $env:PATH
    $probe = Invoke-Recorded 'candidate-direct-repack' (Join-Path $probeBins.candidate 'ciot-arm-repack-check.exe') @()
    if ($probe -notmatch '78/78 direct repacked cases passed') { throw 'Direct repacked test cases did not all execute.' }
} finally { $env:PATH = $savedPath }
if ($Stage -eq 'Small') {
    foreach ($mode in @('stock','candidate')) {
        $text = Invoke-Recorded "$mode-backend-ops" (Join-Path $bins[$mode] 'test-backend-ops.exe') @('test','-b','CPU','-o','MUL_MAT','-p','type_a=pq2_0','-j','1')
        if ($text -notmatch '[1-9][0-9]*/[1-9][0-9]* tests passed') { throw 'No executed backend tests were reported.' }
    }
}
$modelName = if ($Stage -eq 'Large') { 'Ternary-Bonsai-2-27B-PQ2_0.gguf' } else { 'Qwen3-0.6B-PQ2_0.gguf' }
$model = Join-Path $root "models/$modelName"
if (-not (Test-Path $model)) { throw "Missing model: $model" }
$inputFile = Join-Path $run 'ppl-input.txt'
[IO.File]::WriteAllText($inputFile, [IO.File]::ReadAllText((Join-Path $root 'tools/ppl-input.txt')), $utf8)
$ppl = @{}
foreach ($mode in @('stock','candidate')) {
    $ppl[$mode] = Invoke-Recorded "$mode-perplexity" (Join-Path $bins[$mode] 'llama-perplexity.exe') @('-m',$model,'-f',$inputFile,'-c','128','--chunks','3','-b','128','-ub','128','-t',"$Threads",'-tb',"$Threads",'-ngl','0','--prio','2','--load-mode','none')
    Start-Sleep -Seconds 10
}
$stockPpl = [regex]::Matches($ppl.stock, '\[\d+\]([0-9]+\.[0-9]+)') | ForEach-Object Value
$candidatePpl = [regex]::Matches($ppl.candidate, '\[\d+\]([0-9]+\.[0-9]+)') | ForEach-Object Value
if (@($stockPpl).Count -ne 3 -or @($candidatePpl).Count -ne 3 -or ($stockPpl -join ',') -ne ($candidatePpl -join ',')) {
    throw 'Perplexity chunks differ or could not be parsed. Inspect logs before making an exactness claim.'
}
Invoke-Recorded 'greedy-token-check' $Python @((Join-Path $root 'tools/check_arm_tokens.py'),
    '--stock-tool',(Join-Path $probeBins.stock 'ciot-token-check.exe'),'--candidate-tool',(Join-Path $probeBins.candidate 'ciot-token-check.exe'),
    '--model',$model,'--threads',"$Threads",'--out',$run) | Out-Null
foreach ($round in 1..2) {
    $order = if ($round -eq 1) { @('stock','candidate') } else { @('candidate','stock') }
    foreach ($mode in $order) {
        Invoke-Recorded "$mode-bench-round-$round" (Join-Path $bins[$mode] 'llama-bench.exe') @('-m',$model,'-p','64','-n','32','-r','2','-t',"$Threads",'-ngl','0','--prio','2','--load-mode','none','-o','json') | Out-Null
        Start-Sleep -Seconds 10
    }
}
[IO.File]::WriteAllText((Join-Path $run 'checks.json'), (@{ppl_printed_chunks_equal=$true; greedy_token_ids_checked=$true; full_ci_run=$false; repacked_operator_boundary_cases_checked=$true; note='78 direct CPU_REPACK cases cover integer activations with exact Q8_K conversion. Token comparison is limited to three prompts. Standard backend tests use ordinary CPU buffers.'} | ConvertTo-Json), $utf8)
Write-Host "Checks recorded: $run. Inspect logs and remaining checks before submission."
