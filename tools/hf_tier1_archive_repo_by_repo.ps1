param(
    [string]$RepoRoot = "I:\whisper-acft",
    [string]$RunsRoot = "I:\",
    [string]$CacheDir = "I:\hf_model_cache",
    [string]$InputCsv = "",
    [int]$MinSizeMB = 50,
    [switch]$CleanupLocal,
    [switch]$StopOnError
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Read-HfToken([string]$RepoRootPath) {
    if ($env:HF_TOKEN) { return $env:HF_TOKEN.Trim() }
    if ($env:HUGGINGFACE_TOKEN) { return $env:HUGGINGFACE_TOKEN.Trim() }
    $envFile = Join-Path $RepoRootPath ".env"
    if (-not (Test-Path -LiteralPath $envFile)) {
        return ""
    }
    foreach ($raw in Get-Content -Path $envFile -Encoding UTF8) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            continue
        }
        $parts = $line.Split("=", 2)
        $key = $parts[0].Trim()
        if ($key -notin @("HF_TOKEN", "HUGGINGFACE_TOKEN")) {
            continue
        }
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($value) { return $value }
    }
    return ""
}

function Get-LatestDryRunCsv([string]$RepoRootPath) {
    $reportDir = Join-Path $RepoRootPath "hf_tier1_reports"
    if (-not (Test-Path -LiteralPath $reportDir)) {
        throw "hf_tier1_reports does not exist: $reportDir"
    }
    $latest = Get-ChildItem -Path $reportDir -File -Filter "hf_tier1_archive_*.csv" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latest) {
        throw "No report CSV found under $reportDir"
    }
    return $latest.FullName
}

function Get-ModelFilesForRun([string]$RunPath, [long]$MinBytes) {
    $extSet = @(".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".ggml", ".onnx")
    if (-not (Test-Path -LiteralPath $RunPath)) {
        return @()
    }
    $items = Get-ChildItem -Path $RunPath -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $extSet -contains $_.Extension.ToLowerInvariant() -and $_.Length -ge $MinBytes }
    return @($items)
}

function New-PointerFiles(
    [string]$RunPath,
    [string]$RunFolder,
    [string]$RepoId,
    [string]$Revision,
    [string]$CacheDirPath,
    [array]$ModelFiles,
    [array]$DeletedRelPaths,
    [int]$MinSize
) {
    $pointerPath = Join-Path $RunPath "MODEL_POINTER.json"
    $readmePath = Join-Path $RunPath "MODEL_POINTER_README.md"
    $urlPath = Join-Path $RunPath "HF_MODEL_REPO.url"

    $archivedFiles = @()
    foreach ($f in $ModelFiles) {
        $rel = $f.FullName.Substring($RunPath.Length).TrimStart('\') -replace '\\','/'
        $archivedFiles += [pscustomobject]@{
            abs_path = $f.FullName
            rel_path = $rel
            size_bytes = [int64]$f.Length
        }
    }

    $pointer = [pscustomobject]@{
        pointer_version = "1.0"
        created_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        run_folder = $RunFolder
        local_run_path = $RunPath
        repo_id = $RepoId
        repo_url = "https://huggingface.co/$RepoId"
        repo_type = "model"
        private = $true
        revision = $Revision
        cache_strategy = [pscustomobject]@{
            mode = "hybrid"
            central_cache_dir = $CacheDirPath
            local_restore_dir_default = (Join-Path $RunPath "restored_model")
        }
        archive_policy = [pscustomobject]@{
            min_size_mb = $MinSize
            cleanup_local_enabled = [bool]$CleanupLocal
            extensions = @(".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".ggml", ".onnx")
        }
        archived_files = $archivedFiles
        deleted_local_files = $DeletedRelPaths
    }
    $pointer | ConvertTo-Json -Depth 8 | Set-Content -Path $pointerPath -Encoding UTF8

    $restoreCmd = 'python "I:\whisper-acft\tools\hf_tier1_restore_from_pointer.py" --pointer "' + $pointerPath + '" --cache-dir "' + $CacheDirPath + '" --local-dir "' + (Join-Path $RunPath "restored_model") + '"'
    $readmeText = @(
        "# Model Pointer",
        "",
        "- Repo: https://huggingface.co/$RepoId",
        "- Revision: $Revision",
        "- Privacy: private",
        "- Pointer JSON: MODEL_POINTER.json",
        "",
        "## Restore (hybrid cache)",
        "",
        '```powershell',
        $restoreCmd,
        '```'
    ) -join [Environment]::NewLine
    Set-Content -Path $readmePath -Value $readmeText -Encoding UTF8

    @"
[InternetShortcut]
URL=https://huggingface.co/$RepoId
"@ | Set-Content -Path $urlPath -Encoding UTF8
}

if (-not $InputCsv) {
    $InputCsv = Get-LatestDryRunCsv -RepoRootPath $RepoRoot
}
if (-not (Test-Path -LiteralPath $InputCsv)) {
    throw "Input CSV not found: $InputCsv"
}

$hfToken = Read-HfToken -RepoRootPath $RepoRoot
if (-not $hfToken) {
    throw "HF token not found in env or .env"
}
$env:HF_TOKEN = $hfToken
$username = (hf auth whoami).Trim()
if (-not $username) {
    throw "Failed to resolve HF username via hf auth whoami"
}

$rows = @(Import-Csv -Path $InputCsv | Where-Object { $_.status -eq "dry_run" -and $_.repo_id })
$session = Get-Date -Format "yyyyMMdd_HHmmss"
$reportDir = Join-Path $RepoRoot "hf_tier1_reports"
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null

$results = New-Object System.Collections.Generic.List[object]
$minBytes = [int64]$MinSizeMB * 1024 * 1024

$i = 0
foreach ($row in $rows) {
    $i++
    $runFolder = $row.run_folder
    $repoId = $row.repo_id
    $runPath = Join-Path $RunsRoot $runFolder
    $status = "unknown"
    $note = ""
    $fileCount = 0
    $uploadBytes = [int64]0
    $deletedBytes = [int64]0
    $revision = ""

    try {
        Write-Host ("[{0}/{1}] {2}" -f $i, $rows.Count, $runFolder)
        if (-not (Test-Path -LiteralPath $runPath)) {
            $status = "missing_run_folder"
            $note = $runPath
            throw "Run folder missing: $runPath"
        }

        $files = @(Get-ModelFilesForRun -RunPath $runPath -MinBytes $minBytes)
        $fileCount = @($files).Count
        if ($fileCount -gt 0) {
            $measure = $files | Measure-Object -Property Length -Sum
            $uploadBytes = [int64]($measure.Sum)
        } else {
            $uploadBytes = [int64]0
        }

        if ($fileCount -eq 0) {
            $status = "skip_no_files"
            $note = "No model files >= $MinSizeMB MB"
            try {
                $apiUri = "https://huggingface.co/api/models/$repoId"
                $headers = @{ Authorization = "Bearer $hfToken" }
                $repoInfo = Invoke-RestMethod -Uri $apiUri -Headers $headers -Method Get
                $revision = [string]$repoInfo.sha
                $sibCount = @($repoInfo.siblings).Count
                if ($sibCount -gt 1) {
                    New-PointerFiles -RunPath $runPath -RunFolder $runFolder -RepoId $repoId -Revision ($revision ? $revision : "main") -CacheDirPath $CacheDir -ModelFiles @() -DeletedRelPaths @() -MinSize $MinSizeMB
                    $status = "pointer_only_no_local_files"
                    $note = "Local model files absent; pointer generated from existing remote repo"
                }
            } catch {
                $note = "No local files and remote repo not usable: $($_.Exception.Message)"
            }
            $results.Add([pscustomobject]@{
                run_folder = $runFolder
                repo_id = $repoId
                status = $status
                file_count = $fileCount
                upload_bytes = $uploadBytes
                deleted_bytes = $deletedBytes
                revision = $revision
                note = $note
            })
            continue
        }

        hf repo create $repoId --private --exist-ok | Out-Host
        foreach ($f in $files) {
            $rel = $f.FullName.Substring($runPath.Length).TrimStart('\') -replace '\\','/'
            hf upload $repoId $f.FullName $rel --repo-type model --commit-message "Archive $runFolder : $rel" | Out-Host
        }

        $apiUri = "https://huggingface.co/api/models/$repoId"
        $headers = @{ Authorization = "Bearer $hfToken" }
        $repoInfo = Invoke-RestMethod -Uri $apiUri -Headers $headers -Method Get
        $revision = [string]$repoInfo.sha
        $remoteFiles = @{}
        foreach ($s in $repoInfo.siblings) {
            if ($s.rfilename) { $remoteFiles[$s.rfilename] = $true }
        }
        $missing = @()
        foreach ($f in $files) {
            $rel = $f.FullName.Substring($runPath.Length).TrimStart('\') -replace '\\','/'
            if (-not $remoteFiles.ContainsKey($rel)) {
                $missing += $rel
            }
        }
        if ($missing.Count -gt 0) {
            $status = "verify_failed"
            $note = "Missing on remote: $($missing.Count)"
            throw "Remote verification failed for $runFolder"
        }

        $deletedRel = @()
        if ($CleanupLocal) {
            foreach ($f in $files) {
                if (Test-Path -LiteralPath $f.FullName) {
                    Remove-Item -LiteralPath $f.FullName -Force
                    $deletedBytes += [int64]$f.Length
                    $deletedRel += ($f.FullName.Substring($runPath.Length).TrimStart('\') -replace '\\','/')
                }
            }
        }

        New-PointerFiles -RunPath $runPath -RunFolder $runFolder -RepoId $repoId -Revision ($revision ? $revision : "main") -CacheDirPath $CacheDir -ModelFiles $files -DeletedRelPaths $deletedRel -MinSize $MinSizeMB
        $status = "uploaded"
    } catch {
        if ($status -eq "unknown") {
            $status = "error"
            $note = $_.Exception.Message
        } elseif (-not $note) {
            $note = $_.Exception.Message
        }
        Write-Warning ("Failed run {0}: {1}" -f $runFolder, $note)
        if ($StopOnError) {
            $results.Add([pscustomobject]@{
                run_folder = $runFolder
                repo_id = $repoId
                status = $status
                file_count = $fileCount
                upload_bytes = $uploadBytes
                deleted_bytes = $deletedBytes
                revision = $revision
                note = $note
            })
            break
        }
    }

    $results.Add([pscustomobject]@{
        run_folder = $runFolder
        repo_id = $repoId
        status = $status
        file_count = $fileCount
        upload_bytes = $uploadBytes
        deleted_bytes = $deletedBytes
        revision = $revision
        note = $note
    })
}

$outCsv = Join-Path $reportDir ("hf_tier1_repo_by_repo_" + $session + ".csv")
$outJson = Join-Path $reportDir ("hf_tier1_repo_by_repo_" + $session + ".json")
$results | Export-Csv -Path $outCsv -NoTypeInformation -Encoding UTF8

$summary = [pscustomobject]@{
    created_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    input_csv = $InputCsv
    repo_root = $RepoRoot
    runs_root = $RunsRoot
    min_size_mb = $MinSizeMB
    cleanup_local = [bool]$CleanupLocal
    total_runs = $rows.Count
    uploaded_runs = (@($results | Where-Object { $_.status -eq "uploaded" })).Count
    failed_runs = (@($results | Where-Object { $_.status -in @("error", "verify_failed", "missing_run_folder") })).Count
    bytes_deleted = [int64](($results | Measure-Object -Property deleted_bytes -Sum).Sum)
    results = $results
}
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $outJson -Encoding UTF8

$registryPath = Join-Path $RepoRoot "HF_PRIVATE_TIER1_REGISTRY.json"
$existing = @{}
if (Test-Path -LiteralPath $registryPath) {
    try {
        $existing = Get-Content -Path $registryPath -Raw -Encoding UTF8 | ConvertFrom-Json -AsHashtable
    } catch {
        $existing = @{}
    }
}
if (-not $existing.ContainsKey("entries")) {
    $existing["entries"] = @()
}
$map = @{}
foreach ($entry in $existing["entries"]) {
    if ($entry.run_folder) { $map[$entry.run_folder] = $entry }
}
foreach ($r in $results) {
    if (-not $r.repo_id) { continue }
    $map[$r.run_folder] = [ordered]@{
        run_folder = $r.run_folder
        repo_id = $r.repo_id
        status = $r.status
        revision = $r.revision
        updated_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    }
}
$existing["updated_at_utc"] = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$existing["repo_prefix"] = "Whisper-acft"
$existing["username"] = $username
$existing["min_size_mb"] = $MinSizeMB
$existing["entries"] = @($map.Values | Sort-Object run_folder)
$existing["latest_report_json"] = $outJson
$existing["latest_report_csv"] = $outCsv

if (-not $existing.ContainsKey("report_history")) {
    $existing["report_history"] = @()
}
$existing["report_history"] = @($existing["report_history"] + @([ordered]@{
    created_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    json = $outJson
    csv = $outCsv
    cleanup_local = [bool]$CleanupLocal
})) | Select-Object -Last 200

$existing | ConvertTo-Json -Depth 10 | Set-Content -Path $registryPath -Encoding UTF8

Write-Host "Done"
Write-Host "CSV: $outCsv"
Write-Host "JSON: $outJson"
Write-Host "Registry: $registryPath"
$results | Group-Object status | Sort-Object Name | ForEach-Object { Write-Host ("{0}: {1}" -f $_.Name, $_.Count) }
