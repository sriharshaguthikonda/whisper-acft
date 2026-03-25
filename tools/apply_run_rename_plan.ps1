param(
    [string]$PlanCsv = "i:\\whisper-acft\\run_folder_rename_plan.csv",
    [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Junction([string]$LinkPath, [string]$TargetPath) {
    if (Test-Path -LiteralPath $LinkPath) {
        $item = Get-Item -LiteralPath $LinkPath -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            return
        }
        throw "Cannot create junction. Path already exists and is not a junction: $LinkPath"
    }
    cmd /c mklink /J "$LinkPath" "$TargetPath" | Out-Null
}

function Write-RenameNote(
    [string]$FolderPath,
    [pscustomobject]$Row,
    [string]$OldPath,
    [string]$NewPath
) {
    $notePath = Join-Path $FolderPath "RENAMED_FOLDER_NOTE.md"
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $content = @"
# Folder Rename Note

- Timestamp: $ts
- Old path: $OldPath
- New path: $NewPath
- Original name: $($Row.original_name)
- Canonical name: $($Row.new_folder_name)

## Why this was renamed
$($Row.reason)

## Evidence used
$($Row.evidence)

## Canonical tags
- kind: $($Row.kind)
- stage: $($Row.stage)
- base: $($Row.base)
- method: $($Row.method)
- adapter: $($Row.adapter)
- quant: $($Row.quant)
- ctx: $($Row.ctx)
- rows: $($Row.rows)
- run_id: $($Row.run_id)

## Backward compatibility
The old folder name is preserved as a junction alias to this folder.
"@
    Set-Content -Path $notePath -Value $content -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $PlanCsv)) {
    throw "Plan file not found: $PlanCsv"
}

$plan = Import-Csv -Path $PlanCsv
if (-not $plan -or $plan.Count -eq 0) {
    throw "Plan is empty: $PlanCsv"
}

$result = New-Object System.Collections.Generic.List[object]

foreach ($row in $plan) {
    $oldPath = $row.original_path
    $parent = Split-Path -Path $oldPath -Parent
    $newPath = Join-Path $parent $row.new_folder_name
    $status = "dry_run"
    $details = ""

    try {
        $oldExists = Test-Path -LiteralPath $oldPath
        $newExists = Test-Path -LiteralPath $newPath

        if (-not $Apply) {
            if (-not $oldExists -and -not $newExists) {
                $status = "missing_both"
                $details = "old/new paths both missing"
            } elseif ($oldExists -and $newExists) {
                $status = "conflict"
                $details = "old and new both exist"
            } elseif (-not $oldExists -and $newExists) {
                $status = "already_renamed"
                $details = "new exists; old missing"
            } else {
                $status = "ready"
                $details = "rename + junction + note will be applied"
            }
        } else {
            if ($oldExists -and -not $newExists) {
                Rename-Item -LiteralPath $oldPath -NewName $row.new_folder_name
                $status = "renamed"
                $details = "folder renamed"
            } elseif (-not $oldExists -and $newExists) {
                $status = "already_renamed"
                $details = "new exists; old missing"
            } elseif ($oldExists -and $newExists) {
                $oldItem = Get-Item -LiteralPath $oldPath -Force
                if ($oldItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                    $status = "already_renamed"
                    $details = "new exists and old is junction alias"
                } else {
                    $status = "conflict"
                    $details = "old and new both exist"
                }
            } else {
                $status = "missing_both"
                $details = "old/new paths both missing"
            }

            if ($status -in @("renamed", "already_renamed")) {
                Ensure-Junction -LinkPath $oldPath -TargetPath $newPath
                if (Test-Path -LiteralPath $newPath) {
                    Write-RenameNote -FolderPath $newPath -Row $row -OldPath $oldPath -NewPath $newPath
                }
            }
        }
    } catch {
        $status = "error"
        $details = $_.Exception.Message
    }

    $result.Add([pscustomobject]@{
        original_path = $oldPath
        new_path = $newPath
        status = $status
        details = $details
    })
}

$out = [System.IO.Path]::ChangeExtension($PlanCsv, ".apply_result.csv")
$result | Export-Csv -Path $out -NoTypeInformation -Encoding UTF8
Write-Host "Result written:" $out
$result | Group-Object status | Sort-Object Name | ForEach-Object { Write-Host ("{0}: {1}" -f $_.Name, $_.Count) }
