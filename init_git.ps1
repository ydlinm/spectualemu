$ErrorActionPreference = 'Stop'

function Write-Step {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    Write-Step ("Running: git " + ($Args -join ' '))
    & git @Args
    if ($LASTEXITCODE -ne 0) {
        throw ("Git command failed: git " + ($Args -join ' '))
    }
}

function Ensure-GitIdentity {
    Write-Step "Checking Git author identity..."

    $name = (& git config --get user.name)
    if ($LASTEXITCODE -ne 0) {
        $name = $null
    }

    $email = (& git config --get user.email)
    if ($LASTEXITCODE -ne 0) {
        $email = $null
    }

    if ([string]::IsNullOrWhiteSpace($name)) {
        Write-Warn "Git user.name is not configured."
        $name = Read-Host "Enter Git user.name for this repository"
        if ([string]::IsNullOrWhiteSpace($name)) {
            throw "Git user.name cannot be empty."
        }
        Invoke-Git -Args @('config', 'user.name', $name)
    }

    if ([string]::IsNullOrWhiteSpace($email)) {
        Write-Warn "Git user.email is not configured."
        $email = Read-Host "Enter Git user.email for this repository"
        if ([string]::IsNullOrWhiteSpace($email)) {
            throw "Git user.email cannot be empty."
        }
        Invoke-Git -Args @('config', 'user.email', $email)
    }

    Write-Success "Git identity is ready: $name <$email>"
}

try {
    Write-Step "Starting Git initialization workflow in: $(Get-Location)"

    # 1) Rename output -> output_ignored_by_git when present
    $oldFolder = Join-Path (Get-Location) 'output'
    $newFolder = Join-Path (Get-Location) 'output_ignored_by_git'

    if (Test-Path -LiteralPath $oldFolder -PathType Container) {
        if (Test-Path -LiteralPath $newFolder -PathType Container) {
            Write-Warn "Both 'output' and 'output_ignored_by_git' exist. Skipping rename to avoid overwrite."
        }
        else {
            Write-Step "Renaming folder 'output' to 'output_ignored_by_git'..."
            Rename-Item -LiteralPath $oldFolder -NewName 'output_ignored_by_git'
            Write-Success "Folder renamed successfully."
        }
    }
    else {
        Write-Warn "Folder 'output' not found. Skipping rename step."
    }

    # 2) Generate .gitignore
    Write-Step "Creating .gitignore..."
    $gitignoreContent = @'
output_ignored_by_git/
__pycache__/
*.npz
*.jpg
*.png
.DS_Store
dataset/
'@

    Set-Content -Path '.gitignore' -Value $gitignoreContent -Encoding UTF8
    Write-Success ".gitignore created/updated."

    # Check git availability
    Write-Step "Checking if Git is installed..."
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if (-not $gitCmd) {
        throw "Git is not installed or not in PATH. Install Git and rerun this script."
    }
    Write-Success "Git detected: $($gitCmd.Source)"

    # 3) Git initialization and first commit
    Invoke-Git -Args @('init')
    Ensure-GitIdentity
    Invoke-Git -Args @('add', '.')

    Write-Step "Checking for staged changes before commit..."
    & git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Warn "No changes staged. Skipping commit step."
    }
    elseif ($LASTEXITCODE -eq 1) {
        Invoke-Git -Args @('commit', '-m', 'Initial commit: Spectral SSS Algorithm Codebase')
        Write-Success "Initial commit created."
    }
    else {
        throw "Unable to determine staged changes (git diff --cached --quiet failed unexpectedly)."
    }

    Invoke-Git -Args @('branch', '-M', 'main')

    # 4) Prompt for remote URL and push
    $remoteUrl = 'https://github.com/ydlinm/spectualemu.git'

    Write-Step "Configuring remote 'origin'..."
    $existingRemotes = (& git remote)
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to list existing git remotes."
    }

    if ($existingRemotes -contains 'origin') {
        Write-Warn "Remote 'origin' already exists. Updating URL instead of adding."
        Invoke-Git -Args @('remote', 'set-url', 'origin', $remoteUrl)
    }
    else {
        Invoke-Git -Args @('remote', 'add', 'origin', $remoteUrl)
    }
    Write-Success "Remote origin configured: $remoteUrl"

    Invoke-Git -Args @('push', '-u', 'origin', 'main')
    Write-Success "Push completed successfully."

    Write-Success "All done."
}
catch {
    Write-Fail $_.Exception.Message
    exit 1
}


