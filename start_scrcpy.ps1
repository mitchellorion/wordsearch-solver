# Launch scrcpy (color stream is expected — phone grayscale does not pass through).
# For a grayscale high-contrast VIEW matching the bot, also run:
#   python scrcpy_gray_view.py

function Find-Scrcpy {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\scrcpy\scrcpy.exe",
        "$env:LOCALAPPDATA\scrcpy\scrcpy.exe",
        "$env:ProgramFiles\scrcpy\scrcpy.exe",
        "${env:ProgramFiles(x86)}\scrcpy\scrcpy.exe"
    )

    foreach ($p in $candidates) {
        if ($p -and (Test-Path -LiteralPath $p)) { return $p }
    }

    # winget installs under Local\Microsoft\WinGet\Packages\Genymobile.scrcpy_*
    $wingetRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path -LiteralPath $wingetRoot) {
        $hit = Get-ChildItem -Path $wingetRoot -Filter "scrcpy.exe" -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match 'scrcpy' } |
            Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }

    # PATH / App Paths
    $cmd = Get-Command scrcpy.exe -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }

    $cmd = Get-Command scrcpy -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }

    return $null
}

$scrcpy = Find-Scrcpy
if (-not $scrcpy) {
    Write-Error "scrcpy.exe not found. Install via: winget install Genymobile.scrcpy"
    exit 1
}

Write-Host "Using scrcpy: $scrcpy"

# High quality mirror — still COLOR. Filters are applied by the bot on capture.
& $scrcpy `
  --always-on-top `
  --disable-screensaver `
  --stay-awake `
  --video-bit-rate=8M `
  --max-fps=30 `
  --window-title="scrcpy (color stream OK)"
