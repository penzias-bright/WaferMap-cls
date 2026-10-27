$ErrorActionPreference = 'Stop'
$oldPath = $env:PYTHONPATH
$oldUser = $env:PYTHONNOUSERSITE
try {
    $env:PYTHONPATH = ''
    $env:PYTHONNOUSERSITE = '1'
    & 'F:\ProgramData\anaconda3\envs\wafer-custom\python.exe' @args
    $resultCode = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $oldPath
    $env:PYTHONNOUSERSITE = $oldUser
}
exit $resultCode
