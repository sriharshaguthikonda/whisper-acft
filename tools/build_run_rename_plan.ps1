param(
    [string]$Root = "I:\\",
    [string]$OutCsv = "i:\\whisper-acft\\run_folder_rename_plan.csv",
    [string[]]$Paths = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Normalize-Token([string]$value) {
    if ([string]::IsNullOrWhiteSpace($value)) { return "unk" }
    $v = $value.ToLowerInvariant()
    $v = $v -replace "[^a-z0-9]+", "-"
    $v = $v -replace "-{2,}", "-"
    $v = $v.Trim("-")
    if ([string]::IsNullOrWhiteSpace($v)) { return "unk" }
    return $v
}

function First-Hit([string]$path, [string]$pattern) {
    $m = Select-String -Path $path -Pattern $pattern -CaseSensitive:$false -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($m) { return ($m.Line -replace "\s+", " ").Trim() }
    return ""
}

function Parse-Base([string]$line, [string]$name) {
    $s = "$line $name".ToLowerInvariant()
    if ($s -match "futo-org/acft-whisper-small\.en|futo[_-].*small[_-]en|small[_-]en") { return "futo-small-en" }
    if ($s -match "futo-org/acft-whisper-tiny\.en|futo[_-].*tiny[_-]en") { return "futo-tiny-en" }
    if ($s -match "openai/whisper-small\.en|openai[_-].*small[_-]en") { return "openai-small-en" }
    if ($s -match "openai/whisper-tiny\.en|openai[_-].*tiny[_-]en|tiny[_-]en") { return "openai-tiny-en" }
    return "unk"
}

function Parse-Stage([string]$name) {
    if ($name -match "^stage[_-](\d+[a-z]?)") { return $matches[1].ToLowerInvariant() }
    if ($name -match "^dynamic_n_ctx") { return "legacy" }
    if ($name -match "^checkpoints_") { return "legacy" }
    return "unk"
}

function Parse-Method([string]$script, [string]$name, [string]$kind) {
    $s = "$script $name".ToLowerInvariant()
    if ($kind -eq "eval_only") { return "eval-only" }
    if ($s -match "stage_17.*qat_dora") { return "s17-qat-dora" }
    if ($s -match "stage_17.*qat_lora") { return "s17-qat-lora" }
    if ($s -match "stage_17.*_qat\.py") { return "s17-qat-full" }
    if ($s -match "stage_17.*version_only\.py") { return "s17-full" }
    if ($s -match "stage_18_wer_training") { return "s18-full" }
    return "unk"
}

function Parse-Adapter([string]$rLine, [string]$aLine, [string]$doraLine, [string]$script, [int]$adapterFiles, [int]$fullFiles) {
    $r = ""
    $a = ""
    if ($rLine -match "r\s*=\s*(\d+)") { $r = $matches[1] }
    if ($aLine -match "lora_alpha\s*=\s*(\d+)") { $a = $matches[1] }
    $useDora = ($doraLine -match "true")
    $s = $script.ToLowerInvariant()

    if ($useDora -or $s -match "qat_dora") {
        if (-not $r) { $r = "unk" }
        if (-not $a) { $a = "unk" }
        return "dora-r$r-a$a"
    }
    if ($s -match "qat_lora") { return "lora" }
    if ($adapterFiles -gt 0 -and $fullFiles -eq 0) { return "lora" }
    if ($fullFiles -gt 0 -and $adapterFiles -eq 0) { return "full" }
    if ($adapterFiles -gt 0 -and $fullFiles -gt 0) { return "mixed" }
    return "unk"
}

function Parse-Quant([string]$script) {
    $s = $script.ToLowerInvariant()
    if ($s -match "qat") { return "qat" }
    if ([string]::IsNullOrWhiteSpace($s)) { return "unk" }
    return "noqat"
}

function Parse-Ctx([string]$name, [bool]$sawNctx) {
    $n = $name.ToLowerInvariant()
    if ($n -match "dynamic_n_ctx") { return "dyn" }
    if ($n -match "_n_ctx_") { return "static" }
    if ($sawNctx) { return "dyn" }
    return "unk"
}

function Parse-Rows([string]$manifestLine) {
    if ($manifestLine -match "manifest rows:\s*(\d+)") { return $matches[1] }
    return "unk"
}

function Parse-Id([string]$name) {
    if ($name -match "(\d{8})") { return $matches[1] }
    if ($name -match "(\d+)$") { return $matches[1] }
    return "unk"
}

function Build-Name([hashtable]$meta) {
    $tokens = @(
        "RUN",
        ("k-{0}" -f (Normalize-Token $meta["kind"])),
        ("s-{0}" -f (Normalize-Token $meta["stage"])),
        ("b-{0}" -f (Normalize-Token $meta["base"])),
        ("m-{0}" -f (Normalize-Token $meta["method"])),
        ("a-{0}" -f (Normalize-Token $meta["adapter"])),
        ("q-{0}" -f (Normalize-Token $meta["quant"])),
        ("c-{0}" -f (Normalize-Token $meta["ctx"])),
        ("r-{0}" -f (Normalize-Token $meta["rows"])),
        ("id-{0}" -f (Normalize-Token $meta["id"]))
    )
    return ($tokens -join "__")
}

if (-not $Paths -or $Paths.Count -eq 0) {
    $candidates = New-Object System.Collections.Generic.List[string]
    foreach ($f in @("Stage_*", "Dynamic_n_ctx_*", "checkpoints_partialctx*")) {
        Get-ChildItem -Path $Root -Directory -Filter $f -ErrorAction SilentlyContinue | ForEach-Object {
            $candidates.Add($_.FullName)
        }
    }
    $Paths = $candidates | Sort-Object -Unique
}

$rows = New-Object System.Collections.Generic.List[object]
$usedNames = @{}

foreach ($path in $Paths) {
    if (-not (Test-Path -LiteralPath $path)) { continue }
    $item = Get-Item -LiteralPath $path -ErrorAction SilentlyContinue
    if (-not $item -or -not $item.PSIsContainer) { continue }

    $name = $item.Name
    $files = Get-ChildItem -LiteralPath $path -Recurse -File -Depth 4 -ErrorAction SilentlyContinue
    $dirs = Get-ChildItem -LiteralPath $path -Recurse -Directory -Depth 4 -ErrorAction SilentlyContinue
    $logs = Get-ChildItem -LiteralPath $path -Recurse -File -Filter "console.log" -ErrorAction SilentlyContinue

    $runState = @($files | Where-Object Name -eq "run_state.json").Count
    $modelEpoch = @($dirs | Where-Object Name -like "model_epoch_*").Count
    $adapterFiles = @($files | Where-Object Name -eq "adapter_model.safetensors").Count
    $fullFiles = @($files | Where-Object { $_.Name -eq "model.safetensors" -or $_.Name -eq "pytorch_model.bin" }).Count
    $eval = @($files | Where-Object Name -like "evaluation_results*.json").Count
    $pred = @($files | Where-Object Name -like "evaluation_per_sample_predictions*.json").Count

    $script = ""
    $manifest = ""
    $baseLine = ""
    $rLine = ""
    $aLine = ""
    $doraLine = ""
    $sawNctx = $false

    if (@($logs).Count -gt 0) {
        $log = $logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $cmd = First-Hit $log.FullName "^Command:"
        if ($cmd -match "stage_[^\s]+\.py") { $script = $matches[0] }
        $manifest = First-Hit $log.FullName "Manifest rows:"
        $baseLine = First-Hit $log.FullName "--futo-model-id|Loading student from base:"
        $rLine = First-Hit $log.FullName "\[peft\] r="
        $aLine = First-Hit $log.FullName "\[peft\] lora_alpha="
        $doraLine = First-Hit $log.FullName "\[peft\] use_dora="
        $nctxLine = First-Hit $log.FullName "n_ctx="
        $sawNctx = -not [string]::IsNullOrWhiteSpace($nctxLine)
    }

    $kind = "misc"
    if (($eval -gt 0 -or $pred -gt 0) -and ($runState -gt 0 -or $modelEpoch -gt 0 -or $adapterFiles -gt 0 -or $fullFiles -gt 0)) {
        $kind = "train_eval"
    } elseif ($eval -gt 0 -or $pred -gt 0) {
        $kind = "eval_only"
    } elseif ($runState -gt 0 -or $modelEpoch -gt 0 -or $adapterFiles -gt 0 -or $fullFiles -gt 0) {
        $kind = "train_only"
    } elseif (@($logs).Count -gt 0) {
        $kind = "partial"
    }

    $stage = Parse-Stage $name
    $base = Parse-Base $baseLine $name
    $method = Parse-Method $script $name $kind
    $adapter = Parse-Adapter $rLine $aLine $doraLine $script $adapterFiles $fullFiles
    $quant = Parse-Quant $script
    $ctx = Parse-Ctx $name $sawNctx
    $rowsVal = Parse-Rows $manifest
    $id = Parse-Id $name

    $meta = @{
        kind = $kind; stage = $stage; base = $base; method = $method
        adapter = $adapter; quant = $quant; ctx = $ctx; rows = $rowsVal; id = $id
    }
    $newName = Build-Name $meta

    if ($usedNames.ContainsKey($newName)) {
        $usedNames[$newName]++
        $newName = "${newName}__u-$($usedNames[$newName])"
    } else {
        $usedNames[$newName] = 0
    }

    $reason = "renamed using artifact/log-based canonical nomenclature"
    $evidence = @()
    if ($script) { $evidence += "script=$script" }
    if ($manifest) { $evidence += ($manifest -replace "Manifest rows:\s*", "rows=") }
    if ($rLine) { $evidence += ($rLine -replace "^\[peft\]\s*", "") }
    if ($aLine) { $evidence += ($aLine -replace "^\[peft\]\s*", "") }
    if ($doraLine) { $evidence += ($doraLine -replace "^\[peft\]\s*", "") }
    if ($baseLine) { $evidence += "base_hint=$baseLine" }

    $rows.Add([pscustomobject]@{
        original_path = $path
        original_name = $name
        new_folder_name = $newName
        kind = $kind
        stage = $stage
        base = $base
        method = $method
        adapter = $adapter
        quant = $quant
        ctx = $ctx
        rows = $rowsVal
        run_id = $id
        reason = $reason
        evidence = ($evidence -join " | ")
    })
}

$rows | Sort-Object original_path | Export-Csv -Path $OutCsv -NoTypeInformation -Encoding UTF8
Write-Host "Plan written:" $OutCsv
Write-Host "Rows:" $rows.Count
