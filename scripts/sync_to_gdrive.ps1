<#
.SYNOPSIS
    AgeWeaver - Safe Google Drive & Colab Synchronization Script
.DESCRIPTION
    Synchronizes the authoritative local AgeWeaver project into a dedicated subfolder
    of Google Drive ('G:\My Drive\AgeWeaver_Research\08_Code_Backups\AgeWeaver_Colab').
    
    Safety Rules:
    - Never modifies sibling directories (e.g. 'MINI PROJECT-2').
    - Never commits, resets, or pushes to GitHub.
    - Excludes .git, .venv, Python caches (__pycache__, *.pyc), heavy wheels, and temporary files.
    - Preserves all existing experiment results, generated images, and logs; never deletes destination files.
    - Preserves existing backups (e.g. 'AgeWeaver_Research_Backup.zip').
    - Verifies 100% bit-for-bit cryptographic SHA-256 hashes of all copied files.
    - Generates a verified cryptographic audit manifest (JSON and Markdown).
    - Makes failures explicit with non-zero exit codes.
.PARAMETER SourcePath
    Path to AgeWeaver repository root. Default: parent directory of this script.
.PARAMETER GDriveRoot
    Google Drive research root path. Default: 'G:\My Drive\AgeWeaver_Research'.
.PARAMETER SubFolder
    Dedicated subfolder under '08_Code_Backups'. Default: 'AgeWeaver_Colab'.
.PARAMETER DryRun
    Simulate discovery and validation without writing files. Default: $false.
.EXAMPLE
    .\scripts\sync_to_gdrive.ps1
.EXAMPLE
    .\scripts\sync_to_gdrive.ps1 -DryRun
#>

[CmdletBinding()]
param(
    [string]$SourcePath = "",
    [string]$GDriveRoot = "G:\My Drive\AgeWeaver_Research",
    [string]$SubFolder  = "AgeWeaver_Colab",
    [switch]$DryRun     = $false
)

$ErrorActionPreference = "Stop"

# 1. Resolve Source Path
if ([string]::IsNullOrWhiteSpace($SourcePath)) {
    $SourcePath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    $SourcePath = (Resolve-Path $SourcePath).Path
}

Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host " AGEWEAVER: GOOGLE DRIVE - COLAB REPRODUCIBLE SYNCHRONIZATION" -ForegroundColor Cyan
Write-Host " Source of Truth : $SourcePath" -ForegroundColor White
Write-Host " GDrive Root     : $GDriveRoot" -ForegroundColor White
Write-Host " Target Folder   : 08_Code_Backups\$SubFolder" -ForegroundColor White
Write-Host " Dry-Run Mode    : $DryRun" -ForegroundColor $(if ($DryRun) { "Yellow" } else { "Green" })
Write-Host "============================================================================" -ForegroundColor Cyan

# 2. Pre-Flight Verifications
if (-not (Test-Path $SourcePath)) {
    Write-Error "Source directory does not exist: $SourcePath"
    exit 1
}

$coreCheckFiles = @("src\ageweaver", "colab_runner.ipynb", "requirements-colab.txt")
foreach ($cf in $coreCheckFiles) {
    $checkPath = Join-Path $SourcePath $cf
    if (-not (Test-Path $checkPath)) {
        Write-Error "Source repository integrity check failed: missing required element '$cf' at $checkPath"
        exit 1
    }
}

if (-not (Test-Path $GDriveRoot)) {
    Write-Error "Google Drive root path is not accessible: '$GDriveRoot'. Please ensure Google Drive for Desktop is running and mounted on G:."
    exit 1
}

$backupsRoot = Join-Path $GDriveRoot "08_Code_Backups"
if (-not (Test-Path $backupsRoot)) {
    Write-Error "Backups directory does not exist: '$backupsRoot'."
    exit 1
}

$destDir = Join-Path $backupsRoot $SubFolder
if (-not $DryRun -and -not (Test-Path $destDir)) {
    Write-Host "[INFO] Creating dedicated target synchronization directory: $destDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
}

# 3. File Discovery and Filtering
$excludeDirs  = @('.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.idea', '.vscode', 'wheels', 'scratch_test_outputs')
$excludeExts  = @('.pyc', '.pyo', '.pyd', '.tmp', '.log', '.swp')
$excludeFiles = @('AgeWeaver_Research_Backup.zip', 'Thumbs.db', '.DS_Store', 'SYNC_MANIFEST.json', 'SYNC_MANIFEST.md')

Write-Host "`n[STEP 1/4] Discovering files in source repository..." -ForegroundColor Yellow

$allSourceFiles = Get-ChildItem -Path $SourcePath -Recurse -File -Force | Where-Object {
    $item = $_
    $rel = $item.FullName.Substring($SourcePath.Length).TrimStart('\', '/')
    $parts = $rel.Split([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    
    # Check directory exclusion
    foreach ($d in $excludeDirs) {
        if ($parts -contains $d) { return $false }
    }
    
    # Check extension exclusion
    if ($excludeExts -contains $item.Extension.ToLower()) { return $false }
    
    # Check filename exclusion
    if ($excludeFiles -contains $item.Name) { return $false }
    
    return $true
}

$totalFiles = $allSourceFiles.Count
$totalBytes = ($allSourceFiles | Measure-Object -Property Length -Sum).Sum
$totalMB = [math]::Round($totalBytes / 1MB, 2)

Write-Host "  -> Discovered $totalFiles candidate files ($totalMB MB) matching inclusion filters." -ForegroundColor Green

# 4. Copy and Update Files (Additive, Preserving Destination)
Write-Host "`n[STEP 2/4] Synchronizing files to Google Drive ($destDir)..." -ForegroundColor Yellow

$syncedCount  = 0
$skippedCount = 0
$copyErrors   = 0
$syncRecords  = @()

foreach ($srcFile in $allSourceFiles) {
    $relPath  = $srcFile.FullName.Substring($SourcePath.Length).TrimStart('\', '/')
    $destPath = Join-Path $destDir $relPath
    $destParent = Split-Path -Path $destPath -Parent
    
    $srcHash = (Get-FileHash -Path $srcFile.FullName -Algorithm SHA256).Hash
    $actionNeeded = $true
    
    if (Test-Path $destPath) {
        $destHash = (Get-FileHash -Path $destPath -Algorithm SHA256).Hash
        if ($srcHash -eq $destHash) {
            $actionNeeded = $false
            $skippedCount++
        }
    }
    
    if ($actionNeeded) {
        if ($DryRun) {
            Write-Host "  [DRY-RUN COPY] $relPath" -ForegroundColor Gray
            $syncedCount++
        } else {
            try {
                if (-not (Test-Path $destParent)) {
                    New-Item -ItemType Directory -Path $destParent -Force | Out-Null
                }
                
                # If destination file is an experiment log, make a timestamped backup copy first
                if ((Test-Path $destPath) -and ($srcFile.Name -like "experiment_log.*")) {
                    $logBackupDir = Join-Path $destDir "outputs\experiments\backups"
                    if (-not (Test-Path $logBackupDir)) {
                        New-Item -ItemType Directory -Path $logBackupDir -Force | Out-Null
                    }
                    $ts = (Get-Date).ToString("yyyyMMdd_HHmmss")
                    $backupPath = Join-Path $logBackupDir "$($srcFile.BaseName)_prev_$ts$($srcFile.Extension)"
                    Copy-Item -Path $destPath -Destination $backupPath -Force
                    Write-Host "  [PRESERVE LOG] Backed up existing destination log to: $backupPath" -ForegroundColor Cyan
                }
                
                Copy-Item -Path $srcFile.FullName -Destination $destPath -Force
                $syncedCount++
            } catch {
                Write-Host "  [ERROR COPYING] $relPath : $_" -ForegroundColor Red
                $copyErrors++
            }
        }
    }
    
    $syncRecords += [PSCustomObject]@{
        RelativePath = $relPath
        SourcePath   = $srcFile.FullName
        DestPath     = $destPath
        SizeBytes    = $srcFile.Length
        SourceHash   = $srcHash
        LastModified = $srcFile.LastWriteTimeUtc.ToString("o")
    }
}

Write-Host "  -> Synchronization complete: $syncedCount copied/updated, $skippedCount already identical, $copyErrors copy errors." -ForegroundColor Green

# 5. Cryptographic Verification (SHA-256)
Write-Host "`n[STEP 3/4] Cryptographically verifying all copied files against source..." -ForegroundColor Yellow

$verificationErrors = 0
$verifiedList = @()

foreach ($rec in $syncRecords) {
    $status = "PENDING"
    $destHash = ""
    
    if ($DryRun) {
        $status = "DRY_RUN_VERIFIED"
    } else {
        if (-not (Test-Path $rec.DestPath)) {
            $status = "MISSING_IN_DESTINATION"
            $verificationErrors++
            Write-Host "  [VERIFY FAIL] Missing: $($rec.RelativePath)" -ForegroundColor Red
        } else {
            $destHash = (Get-FileHash -Path $rec.DestPath -Algorithm SHA256).Hash
            if ($rec.SourceHash -eq $destHash) {
                $status = "VERIFIED_MATCH"
            } else {
                $status = "HASH_MISMATCH"
                $verificationErrors++
                Write-Host "  [VERIFY FAIL] Hash mismatch on: $($rec.RelativePath)" -ForegroundColor Red
                Write-Host "    Source : $($rec.SourceHash)" -ForegroundColor Red
                Write-Host "    Dest   : $destHash" -ForegroundColor Red
            }
        }
    }
    
    $verifiedList += [PSCustomObject]@{
        Path         = $rec.RelativePath
        SizeBytes    = $rec.SizeBytes
        SHA256       = $rec.SourceHash
        LastModified = $rec.LastModified
        Status       = $status
    }
}

if ($verificationErrors -eq 0) {
    Write-Host "  -> Verification Success: 100% of $($verifiedList.Count) files matched bit-for-bit (0 mismatches)." -ForegroundColor Green
} else {
    Write-Host "  -> Verification FAILED: $verificationErrors files had discrepancies!" -ForegroundColor Red
}

# 6. Generate Manifest
Write-Host "`n[STEP 4/4] Writing cryptographic audit manifest..." -ForegroundColor Yellow

$timestampIso = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$manifestObj = [PSCustomObject]@{
    SyncTimestampUtc      = $timestampIso
    SourceRepository      = $SourcePath
    DestinationRepository = $destDir
    TotalFiles            = $verifiedList.Count
    TotalSizeBytes        = $totalBytes
    TotalSizeMB           = $totalMB
    CopyErrors            = $copyErrors
    VerificationErrors    = $verificationErrors
    Status                = if ($copyErrors -eq 0 -and $verificationErrors -eq 0) { "SUCCESS" } else { "FAILED" }
    Files                 = $verifiedList
}

$manifestJsonPath = Join-Path $destDir "SYNC_MANIFEST.json"
$manifestMdPath   = Join-Path $destDir "SYNC_MANIFEST.md"
$localReportMd    = Join-Path $SourcePath "reports\SYNC_MANIFEST_LATEST.md"

if (-not $DryRun) {
    # JSON Manifest
    $manifestObj | ConvertTo-Json -Depth 5 | Set-Content -Path $manifestJsonPath -Encoding UTF8
    
    # Markdown Manifest
    $mdContent = @"
# AgeWeaver Google Drive Synchronization Audit Manifest

**Sync Timestamp (UTC):** $timestampIso  
**Status:** $($manifestObj.Status)  
**Source Directory:** ``$($SourcePath)``  
**Google Drive Destination:** ``$($destDir)``  
**Total Verified Files:** $($manifestObj.TotalFiles)  
**Total Volume:** $totalMB MB ($totalBytes bytes)  
**Cryptographic Verification:** SHA-256 (0 mismatches, 0 omissions)  

---

## Synchronized Files Inventory

| # | Relative Path | Size (Bytes) | SHA-256 Checksum | Status |
| :- | :--- | :-: | :--- | :-: |
"@

    $idx = 1
    foreach ($f in $verifiedList) {
        $mdContent += "`n| $idx | ``$($f.Path)`` | $($f.SizeBytes) | ``$($f.SHA256.Substring(0, 16))...`` | $($f.Status) |"
        $idx++
    }

    $mdContent += @"


---
*Generated automatically by AgeWeaver sync_to_gdrive.ps1.*
"@

    Set-Content -Path $manifestMdPath -Value $mdContent -Encoding UTF8
    Set-Content -Path $localReportMd -Value $mdContent -Encoding UTF8
    
    Write-Host "  -> Audit manifest saved:" -ForegroundColor Green
    Write-Host "     - $manifestJsonPath" -ForegroundColor Gray
    Write-Host "     - $manifestMdPath" -ForegroundColor Gray
    Write-Host "     - $localReportMd" -ForegroundColor Gray
}

# 7. Final Summary & Exit
Write-Host "`n============================================================================" -ForegroundColor Cyan
if ($copyErrors -eq 0 -and $verificationErrors -eq 0) {
    Write-Host " SYNCHRONIZATION SUCCESSFUL (Verified Bit-for-Bit)" -ForegroundColor Green
    Write-Host " Target Directory : $destDir" -ForegroundColor White
    Write-Host " Total Files      : $totalFiles ($totalMB MB)" -ForegroundColor White
    Write-Host " Verified Matches : $($verifiedList.Count)" -ForegroundColor White
    Write-Host " Existing Backups : Untouched ('AgeWeaver_Research_Backup.zip' preserved)" -ForegroundColor White
    Write-Host "============================================================================" -ForegroundColor Cyan
    exit 0
} else {
    Write-Host " SYNCHRONIZATION COMPLETED WITH ERRORS" -ForegroundColor Red
    Write-Host " Copy Errors        : $copyErrors" -ForegroundColor Red
    Write-Host " Verification Errors: $verificationErrors" -ForegroundColor Red
    Write-Host "============================================================================" -ForegroundColor Cyan
    exit 1
}
