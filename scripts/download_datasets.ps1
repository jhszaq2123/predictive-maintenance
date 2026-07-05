param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

$downloadsDir = Join-Path $ProjectRoot "data\downloads"
$rawDir = Join-Path $ProjectRoot "data\raw"

$datasets = @(
    @{
        Key = "ai4i2020"
        Url = "https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
        ArchiveName = "ai4i2020.zip"
        Destination = Join-Path $rawDir "ai4i2020"
    },
    @{
        Key = "secom"
        Url = "https://archive.ics.uci.edu/static/public/179/secom.zip"
        ArchiveName = "secom.zip"
        Destination = Join-Path $rawDir "secom"
    },
    @{
        Key = "c_mapss"
        Url = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
        ArchiveName = "CMAPSSData.zip"
        Destination = Join-Path $rawDir "c_mapss"
    }
)

New-Item -ItemType Directory -Force -Path $downloadsDir | Out-Null

foreach ($dataset in $datasets) {
    $archivePath = Join-Path $downloadsDir $dataset.ArchiveName
    New-Item -ItemType Directory -Force -Path $dataset.Destination | Out-Null

    Write-Host "Downloading $($dataset.Key) from $($dataset.Url)"
    Invoke-WebRequest -Uri $dataset.Url -OutFile $archivePath -UseBasicParsing

    Write-Host "Extracting $($dataset.Key) into $($dataset.Destination)"
    Expand-Archive -Path $archivePath -DestinationPath $dataset.Destination -Force

    if ($dataset.Key -eq "c_mapss") {
        $nestedArchive = Join-Path $dataset.Destination "6. Turbofan Engine Degradation Simulation Data Set\CMAPSSData.zip"
        $nestedDestination = Join-Path $dataset.Destination "CMAPSSData"
        if (Test-Path $nestedArchive) {
            Write-Host "Extracting nested archive for $($dataset.Key) into $nestedDestination"
            Expand-Archive -Path $nestedArchive -DestinationPath $nestedDestination -Force
        }
    }

    if ($dataset.Key -eq "ai4i2020") {
        $canonicalCsv = Join-Path $rawDir "ai4i2020.csv"
        $extractedCsv = Get-ChildItem -Path $dataset.Destination -Recurse -Filter "ai4i2020.csv" | Select-Object -First 1
        if ($extractedCsv) {
            Copy-Item -Path $extractedCsv.FullName -Destination $canonicalCsv -Force
            Write-Host "Copied AI4I CSV to canonical path $canonicalCsv"
        }
    }
}

Write-Host "Dataset download complete."
