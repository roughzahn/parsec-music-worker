Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ComposeFile = Join-Path $PSScriptRoot 'compose.yaml'
$ServiceName = 'parsec-music-worker-gpu'
$ContainerName = 'parsec-music-worker-gpu'
$EnvFile = 'C:\parsec\gpu-worker-prod.env'
$ModelFile = 'C:\parsec\models\630k-audioset-best.pt'
$CacheDir = 'C:\parsec\models\huggingface'
$Image = 'ghcr.io/roughzahn/parsec-music-worker:stable-gpu'

function Write-Section([string]$Message) {
    Write-Host ''
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

try {
    Write-Host 'Parsec GPU Worker - Stable Updater' -ForegroundColor Cyan
    Write-Host "Image: $Image"
    Write-Host 'Pairing and approval remain manual, as before.'

    foreach ($path in @($ComposeFile, $EnvFile, $ModelFile, $CacheDir)) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Required path is missing: $path"
        }
    }

    Write-Section 'Check Docker'
    $dockerVersion = & docker info --format '{{.ServerVersion}}' 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Engine is not available. Start Docker Desktop first. $dockerVersion"
    }
    Write-Host "Docker Engine: $dockerVersion"

    Write-Section 'Validate production configuration'
    & docker compose -f $ComposeFile config --quiet
    if ($LASTEXITCODE -ne 0) {
        throw 'Docker Compose configuration validation failed.'
    }
    Write-Host 'Compose configuration is valid.'

    Write-Section 'Download stable-gpu image'
    Write-Host 'The running container will not be touched unless this pull succeeds.'
    & docker compose -f $ComposeFile pull $ServiceName
    if ($LASTEXITCODE -ne 0) {
        throw 'Image pull failed. The existing container, if any, was left untouched.'
    }
    Write-Host 'Image pull completed successfully.' -ForegroundColor Green

    Write-Section 'Replace GPU Worker container'
    & docker compose -f $ComposeFile up -d --force-recreate --no-build --pull never $ServiceName
    if ($LASTEXITCODE -ne 0) {
        throw 'Docker Compose could not start the new GPU Worker container.'
    }

    Write-Section 'Wait for container startup'
    $running = $false
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        $stateOutput = & docker inspect --format '{{.State.Status}}' $ContainerName 2>$null
        if ($LASTEXITCODE -eq 0 -and "$stateOutput".Trim() -eq 'running') {
            $running = $true
            break
        }
        if ($LASTEXITCODE -eq 0 -and "$stateOutput".Trim() -in @('exited', 'dead')) {
            break
        }
        Start-Sleep -Seconds 3
    }

    if (-not $running) {
        Write-Host 'Container did not reach running state. Recent logs:' -ForegroundColor Red
        & docker logs --tail 40 $ContainerName
        throw 'GPU Worker container startup failed.'
    }
    Write-Host 'Container is running.' -ForegroundColor Green

    Write-Section 'Check worker health and version'
    $health = $null
    $healthDeadline = (Get-Date).AddSeconds(45)
    $healthProbe = @'
import urllib.request
import urllib.error
try:
    response = urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=3)
    print(response.read().decode())
except urllib.error.HTTPError as error:
    print(error.read().decode())
'@

    while ((Get-Date) -lt $healthDeadline) {
        $healthOutput = & docker exec $ContainerName python -c $healthProbe 2>$null
        if ($LASTEXITCODE -eq 0 -and $healthOutput) {
            try {
                $health = ($healthOutput -join [Environment]::NewLine) | ConvertFrom-Json
                break
            }
            catch {
                $health = $null
            }
        }
        Start-Sleep -Seconds 5
    }

    if ($null -ne $health) {
        if ($health.softwareVersion) {
            Write-Host "Worker version: $($health.softwareVersion)"
        }
        if ($health.ready -eq $true) {
            Write-Host 'Worker health: UP - ready for jobs.' -ForegroundColor Green
        }
        else {
            Write-Host 'Container is running, but the worker is not ready yet.' -ForegroundColor Yellow
            Write-Host 'If a new pairing request is shown, approve it in the Parsec Admin Tool.'
            Write-Host 'Model initialization or the first gateway connection may also still be in progress.'
        }
    }
    else {
        Write-Host 'Container is running, but /health did not respond within 45 seconds.' -ForegroundColor Yellow
        Write-Host 'Check the container logs below for the reason.'
    }

    Write-Section 'Recent worker logs'
    & docker logs --tail 20 $ContainerName

    Write-Host ''
    Write-Host 'Updater finished. No host ports were published.' -ForegroundColor Green
}
catch {
    Write-Host ''
    Write-Host "UPDATE FAILED: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
