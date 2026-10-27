$ErrorActionPreference = 'Stop'
$oldPath = $env:PYTHONPATH
$oldUser = $env:PYTHONNOUSERSITE
$oldConfig = $env:YOLO_CONFIG_DIR
try {
    $env:PYTHONPATH = ''
    $env:PYTHONNOUSERSITE = '1'
    $env:YOLO_CONFIG_DIR = Join-Path $PSScriptRoot '../environment/ultralytics'
    & 'F:\ProgramData\anaconda3\envs\wafer-yolo26\python.exe' @args
    $resultCode = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $oldPath
    $env:PYTHONNOUSERSITE = $oldUser
    $env:YOLO_CONFIG_DIR = $oldConfig
}
exit $resultCode
