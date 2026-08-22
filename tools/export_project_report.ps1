param(
    [string]$OutputFile = "project_report.md"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"


# -------------------------------------------------------------------------
# Native command helpers
# -------------------------------------------------------------------------

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @()
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        $commandOutput = @(
            & $Executable @Arguments 2>&1
        )

        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    $textOutput = (
        $commandOutput |
        ForEach-Object { $_.ToString() }
    ) -join "`n"

    return [PSCustomObject]@{
        ExitCode = $exitCode
        Output   = $textOutput
    }
}


function Get-NativeLines {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @()
    )

    $result = Invoke-NativeCommand `
        -Executable $Executable `
        -Arguments $Arguments

    if ($result.ExitCode -ne 0) {
        return @()
    }

    if ([string]::IsNullOrWhiteSpace($result.Output)) {
        return @()
    }

    return @(
        $result.Output -split "\r?\n" |
        Where-Object {
            -not [string]::IsNullOrWhiteSpace($_)
        }
    )
}


# -------------------------------------------------------------------------
# Confirm repository root
# -------------------------------------------------------------------------

$repoResult = Invoke-NativeCommand `
    -Executable "git" `
    -Arguments @("rev-parse", "--show-toplevel")

if ($repoResult.ExitCode -ne 0) {
    throw "This directory is not inside a Git repository."
}

$repoRoot = [System.IO.Path]::GetFullPath(
    $repoResult.Output.Trim().Replace("/", "\")
).TrimEnd("\")

$currentPath = [System.IO.Path]::GetFullPath(
    (Get-Location).Path
).TrimEnd("\")

if (-not [string]::Equals(
    $currentPath,
    $repoRoot,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Run this script from the repository root: $repoRoot"
}


# -------------------------------------------------------------------------
# Resolve main reference
# -------------------------------------------------------------------------

$mainRef = $null

$localMain = Invoke-NativeCommand `
    -Executable "git" `
    -Arguments @(
        "show-ref",
        "--verify",
        "--quiet",
        "refs/heads/main"
    )

if ($localMain.ExitCode -eq 0) {
    $mainRef = "main"
}
else {
    $originMain = Invoke-NativeCommand `
        -Executable "git" `
        -Arguments @(
            "show-ref",
            "--verify",
            "--quiet",
            "refs/remotes/origin/main"
        )

    if ($originMain.ExitCode -eq 0) {
        $mainRef = "origin/main"
    }
}

if (-not $mainRef) {
    throw "Neither main nor origin/main was found."
}


# -------------------------------------------------------------------------
# Repository metadata
# -------------------------------------------------------------------------

$branch = (
    Invoke-NativeCommand `
        -Executable "git" `
        -Arguments @("branch", "--show-current")
).Output.Trim()

$head = (
    Invoke-NativeCommand `
        -Executable "git" `
        -Arguments @("rev-parse", "HEAD")
).Output.Trim()

$mainHash = (
    Invoke-NativeCommand `
        -Executable "git" `
        -Arguments @("rev-parse", $mainRef)
).Output.Trim()

$mergeBase = (
    Invoke-NativeCommand `
        -Executable "git" `
        -Arguments @("merge-base", "HEAD", $mainRef)
).Output.Trim()

$generatedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"

$outputPath = Join-Path $repoRoot $OutputFile


# -------------------------------------------------------------------------
# UTF-8 writer
# -------------------------------------------------------------------------

$utf8Encoding = New-Object System.Text.UTF8Encoding($false)

$writer = New-Object System.IO.StreamWriter(
    $outputPath,
    $false,
    $utf8Encoding
)


function Add-Line {
    param(
        [AllowEmptyString()]
        [string]$Text = ""
    )

    $script:writer.WriteLine($Text)
}


function Add-RawText {
    param(
        [AllowEmptyString()]
        [string]$Text = ""
    )

    if ([string]::IsNullOrEmpty($Text)) {
        return
    }

    foreach ($line in ($Text -split "\r?\n")) {
        Add-Line $line
    }
}


function Add-NativeBlock {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title,

        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @(),

        [Parameter(Mandatory = $false)]
        [int[]]$SuccessExitCodes = @(0)
    )

    Add-Line "## $Title"
    Add-Line ""
    Add-Line '`````text'

    $result = Invoke-NativeCommand `
        -Executable $Executable `
        -Arguments $Arguments

    if ($SuccessExitCodes -notcontains $result.ExitCode) {
        Add-Line "[Command exit code: $($result.ExitCode)]"
        Add-Line ""
    }

    if ([string]::IsNullOrWhiteSpace($result.Output)) {
        Add-Line "(no output)"
    }
    else {
        Add-RawText $result.Output
    }

    Add-Line '`````'
    Add-Line ""
}


function Get-MarkdownLanguage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $extension = [System.IO.Path]::GetExtension($Path).ToLowerInvariant()

    switch ($extension) {
        ".py"   { return "python" }
        ".ps1"  { return "powershell" }
        ".json" { return "json" }
        ".toml" { return "toml" }
        ".yml"  { return "yaml" }
        ".yaml" { return "yaml" }
        ".md"   { return "markdown" }
        default { return "text" }
    }
}


function Add-CurrentFileSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $fullPath = Join-Path $repoRoot $Path

    Add-Line "### $Path"
    Add-Line ""

    if (-not (Test-Path $fullPath)) {
        Add-Line "_File does not exist in the current working tree._"
        Add-Line ""
        return
    }

    $language = Get-MarkdownLanguage -Path $Path

    Add-Line "`````$language"

    try {
        $content = [System.IO.File]::ReadAllText(
            $fullPath,
            [System.Text.Encoding]::UTF8
        )

        Add-RawText $content
    }
    catch {
        Add-Line "[Could not read file as UTF-8 text]"
        Add-Line $_.Exception.Message
    }

    Add-Line '`````'
    Add-Line ""
}


function Add-MainFileSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    Add-Line "### $mainRef : $Path"
    Add-Line ""

    $result = Invoke-NativeCommand `
        -Executable "git" `
        -Arguments @(
            "show",
            "${mainRef}:$Path"
        )

    if ($result.ExitCode -ne 0) {
        Add-Line "_File was not present in $mainRef._"
        Add-Line ""
        return
    }

    $language = Get-MarkdownLanguage -Path $Path

    Add-Line "`````$language"
    Add-RawText $result.Output
    Add-Line '`````'
    Add-Line ""
}


function Add-NotebookSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $fullPath = Join-Path $repoRoot $Path

    Add-Line "### Notebook: $Path"
    Add-Line ""
    Add-Line "> Outputs, images, widget state, and base64 payloads are intentionally excluded."
    Add-Line ""

    if (-not (Test-Path $fullPath)) {
        Add-Line "_Notebook does not exist in the current working tree._"
        Add-Line ""
        return
    }

    try {
        $rawNotebook = [System.IO.File]::ReadAllText(
            $fullPath,
            [System.Text.Encoding]::UTF8
        )

        $notebook = $rawNotebook | ConvertFrom-Json
    }
    catch {
        Add-Line "_Notebook JSON could not be parsed._"
        Add-Line ""
        Add-Line $_.Exception.Message
        Add-Line ""
        return
    }

    $cellIndex = 0

    foreach ($cell in $notebook.cells) {
        $cellType = [string]$cell.cell_type

        Add-Line "#### Cell $cellIndex - $cellType"
        Add-Line ""

        if ($cellType -eq "code") {
            Add-Line '`````python'
        }
        else {
            Add-Line '`````text'
        }

        $sourceText = ""

        if ($null -ne $cell.source) {
            $sourceText = ($cell.source -join "")
        }

        Add-RawText $sourceText

        Add-Line '`````'
        Add-Line ""

        $cellIndex++
    }
}


function Add-SearchBlock {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title,

        [Parameter(Mandatory = $true)]
        [string]$Pattern
    )

    Add-Line "## $Title"
    Add-Line ""
    Add-Line '`````text'

    $arguments = @(
        "grep",
        "--untracked",
        "-n",
        "-i",
        "-E",
        $Pattern,
        "--",
        "*.py",
        "*.md",
        "*.toml",
        "*.yml",
        "*.yaml",
        "*.json",
        "*.ps1",
        ":(exclude)$OutputFile"
    )

    $result = Invoke-NativeCommand `
        -Executable "git" `
        -Arguments $arguments

    if ($result.ExitCode -eq 1) {
        Add-Line "No matches found."
    }
    elseif ($result.ExitCode -ne 0) {
        Add-Line "[git grep exit code: $($result.ExitCode)]"
        Add-RawText $result.Output
    }
    elseif ([string]::IsNullOrWhiteSpace($result.Output)) {
        Add-Line "No matches found."
    }
    else {
        Add-RawText $result.Output
    }

    Add-Line '`````'
    Add-Line ""
}


try {

    # ---------------------------------------------------------------------
    # Header
    # ---------------------------------------------------------------------

    Add-Line "# ET Fundacion - Repository Status Report"
    Add-Line ""
    Add-Line "**Generated:** $generatedAt  "
    Add-Line "**Repository root:** ``$repoRoot``  "
    Add-Line "**Current branch:** ``$branch``  "
    Add-Line "**Current HEAD:** ``$head``  "
    Add-Line "**Comparison reference:** ``$mainRef``  "
    Add-Line "**Main hash:** ``$mainHash``  "
    Add-Line "**Merge base:** ``$mergeBase``"
    Add-Line ""

    Add-Line "> This report records the actual repository and working-tree state."
    Add-Line "> Current implementation is not assumed to be the final methodology."
    Add-Line "> Git line-ending warnings are recorded but are not treated as command failures when Git exits successfully."
    Add-Line "> Notebook outputs and embedded images are excluded deliberately."
    Add-Line ""


    # ---------------------------------------------------------------------
    # Runtime environment
    # ---------------------------------------------------------------------

    Add-Line "## Runtime environment identity"
    Add-Line ""
    Add-Line '`````text'
    Add-Line "CONDA_DEFAULT_ENV = $env:CONDA_DEFAULT_ENV"
    Add-Line "CONDA_PREFIX      = $env:CONDA_PREFIX"

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue

    if ($null -ne $pythonCommand) {
        Add-Line "PowerShell python  = $($pythonCommand.Source)"
    }
    else {
        Add-Line "PowerShell python  = NOT FOUND"
    }

    Add-Line '`````'
    Add-Line ""

    Add-NativeBlock `
        -Title "Python runtime" `
        -Executable "python" `
        -Arguments @(
            "-c",
            "import sys; print('executable:', sys.executable); print('version:', sys.version)"
        )

    Add-NativeBlock `
        -Title "Conda environments" `
        -Executable "conda" `
        -Arguments @("info", "--envs")


    # ---------------------------------------------------------------------
    # Git state
    # ---------------------------------------------------------------------

    Add-NativeBlock `
        -Title "Git status" `
        -Executable "git" `
        -Arguments @(
            "status",
            "--short",
            "--branch",
            "--untracked-files=all"
        )

    Add-NativeBlock `
        -Title "Branches" `
        -Executable "git" `
        -Arguments @("branch", "-vv")

    Add-NativeBlock `
        -Title "Recent commit graph" `
        -Executable "git" `
        -Arguments @(
            "log",
            "--oneline",
            "--decorate",
            "--graph",
            "--all",
            "-n",
            "50"
        )

    Add-NativeBlock `
        -Title "Commits on current branch since main" `
        -Executable "git" `
        -Arguments @(
            "log",
            "--oneline",
            "--decorate",
            "$mainRef..HEAD"
        )


    # ---------------------------------------------------------------------
    # Main versus branch
    # ---------------------------------------------------------------------

    Add-NativeBlock `
        -Title "Committed changes relative to main - summary" `
        -Executable "git" `
        -Arguments @(
            "-c",
            "core.safecrlf=false",
            "diff",
            "--stat",
            "$mainRef...HEAD"
        )

    Add-NativeBlock `
        -Title "Committed changes relative to main - files" `
        -Executable "git" `
        -Arguments @(
            "-c",
            "core.safecrlf=false",
            "diff",
            "--name-status",
            "$mainRef...HEAD"
        )

    Add-NativeBlock `
        -Title "Full current working tree relative to main - summary" `
        -Executable "git" `
        -Arguments @(
            "-c",
            "core.safecrlf=false",
            "diff",
            "--stat",
            $mainRef
        )

    Add-NativeBlock `
        -Title "Full current working tree relative to main - files" `
        -Executable "git" `
        -Arguments @(
            "-c",
            "core.safecrlf=false",
            "diff",
            "--name-status",
            $mainRef
        )

    Add-NativeBlock `
        -Title "Unstaged tracked changes" `
        -Executable "git" `
        -Arguments @(
            "-c",
            "core.safecrlf=false",
            "diff",
            "--stat"
        )

    Add-NativeBlock `
        -Title "Staged changes" `
        -Executable "git" `
        -Arguments @(
            "-c",
            "core.safecrlf=false",
            "diff",
            "--cached",
            "--stat"
        )

    Add-NativeBlock `
        -Title "Untracked files" `
        -Executable "git" `
        -Arguments @(
            "ls-files",
            "--others",
            "--exclude-standard"
        )


    # ---------------------------------------------------------------------
    # Line-ending diagnostics
    # ---------------------------------------------------------------------

    Add-NativeBlock `
        -Title "Git line-ending configuration - autocrlf" `
        -Executable "git" `
        -Arguments @(
            "config",
            "--get",
            "core.autocrlf"
        ) `
        -SuccessExitCodes @(0, 1)

    Add-NativeBlock `
        -Title "Git line-ending configuration - safecrlf" `
        -Executable "git" `
        -Arguments @(
            "config",
            "--get",
            "core.safecrlf"
        ) `
        -SuccessExitCodes @(0, 1)

    Add-NativeBlock `
        -Title "Tracked file line endings" `
        -Executable "git" `
        -Arguments @(
            "ls-files",
            "--eol"
        )


    # ---------------------------------------------------------------------
    # Current repository inventory
    # ---------------------------------------------------------------------

    $trackedFiles = Get-NativeLines `
        -Executable "git" `
        -Arguments @("ls-files")

    $untrackedFiles = Get-NativeLines `
        -Executable "git" `
        -Arguments @(
            "ls-files",
            "--others",
            "--exclude-standard"
        )

    $untrackedFiles = @(
        $untrackedFiles |
        Where-Object {
            $_ -ne $OutputFile
        }
    )

    $allCurrentFiles = @(
        $trackedFiles + $untrackedFiles |
        Sort-Object -Unique
    )

    Add-Line "## Current repository inventory"
    Add-Line ""
    Add-Line '`````text'

    foreach ($path in $allCurrentFiles) {

        if ($trackedFiles -contains $path) {
            Add-Line "[tracked]   $path"
        }
        else {
            Add-Line "[untracked] $path"
        }
    }

    Add-Line '`````'
    Add-Line ""


    # ---------------------------------------------------------------------
    # Full text diff for code/configuration
    # ---------------------------------------------------------------------

    Add-NativeBlock `
        -Title "Full source and configuration diff relative to main" `
        -Executable "git" `
        -Arguments @(
            "-c",
            "core.safecrlf=false",
            "diff",
            "--no-ext-diff",
            "--unified=5",
            $mainRef,
            "--",
            "*.py",
            "*.md",
            "*.toml",
            "*.yml",
            "*.yaml",
            "*.json",
            "*.ps1",
            ":(exclude)$OutputFile"
        )


    # ---------------------------------------------------------------------
    # Scientific configuration searches
    # ---------------------------------------------------------------------

    Add-SearchBlock `
        -Title "MODIS ET and quality-control references" `
        -Pattern "MOD16|MODIS|ET_QC|modis_good|MODIS_REQUIRE|MODIS_STRICT|ET_mm"

    Add-SearchBlock `
        -Title "Sentinel-2 and HLS references" `
        -Pattern "Sentinel-2|sentinel2|HLS|HLSS30|HLSL30|Fmask|medoid|RedEdge|NDRE"

    Add-SearchBlock `
        -Title "FVC and albedo references" `
        -Pattern "FVC|endmember|albedo|NDVI"

    Add-SearchBlock `
        -Title "Sentinel-1 configuration and terrain references" `
        -Pattern "Sentinel-1|sentinel1|COPERNICUS/S1|relativeOrbit|relative_orbit|ASCENDING|DESCENDING|VV|VH|angle|RTC|gamma|sigma"

    Add-SearchBlock `
        -Title "Meteorological and reference-ET references" `
        -Pattern "ERA5|CHIRPS|VPD|Wind|wind_10m|Wind2m|Solar|Radiation|ETo|ETr|reference_et|Penman"

    Add-SearchBlock `
        -Title "LST and thermal references" `
        -Pattern "LST|landsat_lst|TVDI|thermal"

    Add-SearchBlock `
        -Title "Coverage and validity references" `
        -Pattern "coverage|FULL_COVERAGE|valid_observation|stats_complete|meteo_complete|threshold|0\.999|90\.0|80\.0"

    Add-SearchBlock `
        -Title "Spatial support references" `
        -Pattern "ANALYSIS_SCALE|ANALYSIS_CRS|30 m|30m|60 m|60m|500 m|500m|scale|projection|footprint|neighborhood|local_60"

    Add-SearchBlock `
        -Title "Target, features, and leakage references" `
        -Pattern "PREDICTOR|MODEL_FEATURE|FORBIDDEN|target|feature|leakage|ET_mm_day|station_id|modis_pixel"

    Add-SearchBlock `
        -Title "Validation and metrics references" `
        -Pattern "validation|cross.validation|spatial|temporal|block|R2|RMSE|MAE|bias|KGE|baseline|persistence"


    # ---------------------------------------------------------------------
    # Current source snapshots
    # ---------------------------------------------------------------------

    Add-Line "## Current source snapshots"
    Add-Line ""
    Add-Line "> These are the current working-tree files, including untracked source files."
    Add-Line ""

    $snapshotExtensions = @(
        ".py",
        ".ps1",
        ".json",
        ".toml",
        ".yml",
        ".yaml"
    )

    $snapshotPaths = @(
        $allCurrentFiles |
        Where-Object {

            $extension = [System.IO.Path]::GetExtension($_).ToLowerInvariant()

            (
                $_ -like "src/*" -or
                $_ -like "scripts/*" -or
                $_ -like "config/*" -or
                $_ -like "tools/*" -or
                $_ -eq "environment.yml" -or
                $_ -eq "pyproject.toml"
            ) -and
            ($snapshotExtensions -contains $extension)
        } |
        Sort-Object -Unique
    )

    foreach ($path in $snapshotPaths) {
        Add-CurrentFileSnapshot -Path $path
    }


    # ---------------------------------------------------------------------
    # Main snapshots for files changed relative to main
    # ---------------------------------------------------------------------

    $changedAgainstMain = Get-NativeLines `
        -Executable "git" `
        -Arguments @(
            "-c",
            "core.safecrlf=false",
            "diff",
            "--name-only",
            $mainRef,
            "--",
            "*.py",
            "*.ps1",
            "*.json",
            "*.toml",
            "*.yml",
            "*.yaml"
        )

    Add-Line "## Main baseline snapshots for changed tracked files"
    Add-Line ""
    Add-Line "> These snapshots preserve the corresponding main implementation for direct comparison."
    Add-Line ""

    foreach ($path in ($changedAgainstMain | Sort-Object -Unique)) {
        Add-MainFileSnapshot -Path $path
    }


    # ---------------------------------------------------------------------
    # Explicit snapshots of untracked text source files
    # ---------------------------------------------------------------------

    Add-Line "## Untracked source snapshots"
    Add-Line ""
    Add-Line "> These files are not represented by git diff because they are not yet tracked."
    Add-Line ""

    $textExtensions = @(
        ".py",
        ".ps1",
        ".json",
        ".toml",
        ".yml",
        ".yaml",
        ".md"
    )

    $untrackedTextFiles = @(
        $untrackedFiles |
        Where-Object {
            $textExtensions -contains (
                [System.IO.Path]::GetExtension($_).ToLowerInvariant()
            )
        } |
        Sort-Object -Unique
    )

    if ($untrackedTextFiles.Count -eq 0) {
        Add-Line "_No untracked text source files detected._"
        Add-Line ""
    }
    else {
        foreach ($path in $untrackedTextFiles) {
            Add-CurrentFileSnapshot -Path $path
        }
    }


    # ---------------------------------------------------------------------
    # Notebook source snapshots
    # ---------------------------------------------------------------------

    Add-Line "## Notebook source snapshots"
    Add-Line ""
    Add-Line "> Only markdown and code cell sources are exported."
    Add-Line "> Notebook outputs are excluded to prevent base64 images and cached results from dominating the report."
    Add-Line ""

    $notebookDirectory = Join-Path $repoRoot "notebooks"

    if (Test-Path $notebookDirectory) {

        $notebookFiles = @(
            Get-ChildItem `
                -Path $notebookDirectory `
                -Filter "*.ipynb" `
                -File `
                -Recurse |
            ForEach-Object {
                $_.FullName.Substring(
                    $repoRoot.Length + 1
                ).Replace("\", "/")
            } |
            Sort-Object
        )

        if ($notebookFiles.Count -eq 0) {
            Add-Line "_No notebooks found._"
            Add-Line ""
        }
        else {
            foreach ($notebookPath in $notebookFiles) {
                Add-NotebookSnapshot -Path $notebookPath
            }
        }
    }
    else {
        Add-Line "_Notebook directory not found._"
        Add-Line ""
    }


    # ---------------------------------------------------------------------
    # README
    # ---------------------------------------------------------------------

    if (Test-Path (Join-Path $repoRoot "README.md")) {
        Add-Line "## Current README"
        Add-Line ""
        Add-CurrentFileSnapshot -Path "README.md"
    }


    # ---------------------------------------------------------------------
    # Methodological questions
    # ---------------------------------------------------------------------

    Add-Line "## Open methodological questions"
    Add-Line ""

    Add-Line "- MODIS QC: compare physically valid observations against stricter QC subsets without allowing the extraction filter to predetermine model performance."
    Add-Line "- Optical source: compare Sentinel-2, HLS S30, and HLS S30 plus L30 using common support and explicit native spatial resolution."
    Add-Line "- HLS feature compatibility: determine whether red-edge predictors require S30-only processing."
    Add-Line "- Sentinel-1 orbit: preserve the implemented R077 configuration as the reproducible baseline until orbit alternatives are tested on the same footprints and periods."
    Add-Line "- Sentinel-1 terrain treatment: compare the current GEE GRD implementation with a radiometrically terrain-corrected alternative before selecting the definitive radar preprocessing."
    Add-Line "- Sentinel-1 angle: distinguish diagnostic geometry variables from transferable model predictors."
    Add-Line "- Meteorological support: preserve native ERA5-Land and CHIRPS support and distinguish 10 m wind from any derived 2 m wind used by reference ET."
    Add-Line "- Target formulation: compare direct ET prediction against reference-ET-normalized formulations under identical validation folds."
    Add-Line "- Thermal block: treat Landsat LST or TVDI as an ablation experiment rather than a mandatory predictor."
    Add-Line "- Spatial architecture: distinguish MODIS-footprint training support from the final fine-grid prediction support."
    Add-Line "- Leakage control: define explicit model-feature allowlists and forbidden columns before model fitting."
    Add-Line "- Validation: spatial dependence and spatial generalization must be evaluated before interpreting random or temporal cross-validation performance."
    Add-Line "- Baselines: candidate models must be compared against appropriate persistence and climatological baselines."
    Add-Line "- Downscaling conservation: evaluate whether fine-grid predictions preserve the parent MODIS footprint estimate."
    Add-Line "- Area of applicability: predictions outside the training predictor domain must be identified explicitly."
    Add-Line ""


    # ---------------------------------------------------------------------
    # Reproduction entry points
    # ---------------------------------------------------------------------

    Add-Line "## Reproduction entry points"
    Add-Line ""
    Add-Line '`````text'

    foreach ($path in $allCurrentFiles) {
        if (
            $path -like "scripts/*.py" -or
            $path -like "run_*.py"
        ) {
            Add-Line $path
        }
    }

    Add-Line '`````'
    Add-Line ""


    # ---------------------------------------------------------------------
    # Provenance
    # ---------------------------------------------------------------------

    Add-Line "## Report provenance"
    Add-Line ""
    Add-Line "- Repository branch: ``$branch``"
    Add-Line "- HEAD: ``$head``"
    Add-Line "- Compared against: ``$mainRef``"
    Add-Line "- Main hash: ``$mainHash``"
    Add-Line "- Merge base: ``$mergeBase``"
    Add-Line "- Includes tracked and untracked source inventory."
    Add-Line "- Includes current source snapshots."
    Add-Line "- Includes main snapshots for changed tracked source files."
    Add-Line "- Includes notebook source cells but excludes notebook outputs."
    Add-Line "- Git stderr warnings are retained and evaluated using the actual command exit code."
    Add-Line "- Output encoding: UTF-8."
    Add-Line "- Scientific interpretation is intentionally separated from repository inventory."
    Add-Line ""
}
finally {
    $writer.Flush()
    $writer.Close()
}

Write-Host ""
Write-Host "Report created successfully:"
Write-Host "  $outputPath"
Write-Host ""
Write-Host "Branch:"
Write-Host "  $branch"
Write-Host ""
Write-Host "HEAD:"
Write-Host "  $head"
Write-Host ""
Write-Host "Compared against:"
Write-Host "  $mainRef ($mainHash)"
Write-Host ""