$PROJECT_DIR = (Get-Location).Path
$TRAIN_RECORD = Join-Path $PROJECT_DIR "data\tfrecord\coco_train.record"
$VAL_RECORD   = Join-Path $PROJECT_DIR "data\tfrecord\coco_val.record"
$LABEL_MAP    = Join-Path $PROJECT_DIR "data\label_map.pbtxt"
$FINE_TUNE_CKPT = Join-Path $PROJECT_DIR "models\pretrained\mask_rcnn\checkpoint\ckpt-0"

$cfgPath = Join-Path $PROJECT_DIR "scripts\mask_rcnn.config"
$backupPath = Join-Path $PROJECT_DIR "scripts\mask_rcnn.config.bak"

if (-not (Test-Path $cfgPath)) {
  Write-Error "Config not found: $cfgPath"
  exit 1
}


$NUM_CLASSES = 0
if (Test-Path $LABEL_MAP) {
  $NUM_CLASSES = (Select-String -Path $LABEL_MAP -Pattern '^\s*item\s*\{' -AllMatches).Matches.Count
  Write-Host "Detected NUM_CLASSES =" $NUM_CLASSES
} else {
  Write-Warning "Label map not found at $LABEL_MAP. Please set \$LABEL_MAP manually."
}

Copy-Item -Path $cfgPath -Destination $backupPath -Force
Write-Host "Backup saved to $backupPath"

$text = Get-Content -Path $cfgPath -Raw -ErrorAction Stop

if ($NUM_CLASSES -gt 0) {
  $text = [regex]::Replace($text, 'num_classes:\s*\d+', "num_classes: $NUM_CLASSES")
  Write-Host "Replaced num_classes -> $NUM_CLASSES"
} else {
  Write-Host "Skipping num_classes replace (NUM_CLASSES=0)"
}

$text = [regex]::Replace($text, 'batch_size:\s*\d+', 'batch_size: 2')
Write-Host "Set batch_size -> 2 (if present)"

$text = [regex]::Replace($text, 'num_steps:\s*\d+', 'num_steps: 20000')
Write-Host "Set num_steps -> 20000 (if present)"

if (Test-Path $FINE_TUNE_CKPT) {
  if ($text -match 'fine_tune_checkpoint:\s*".*"') {
    $text = [regex]::Replace($text, 'fine_tune_checkpoint:\s*".*"', "fine_tune_checkpoint: `"$FINE_TUNE_CKPT`"")
  } else {
    $text = [regex]::Replace($text, '(?m)^(train_config:\s*\{)', "$1`n  fine_tune_checkpoint: `"$FINE_TUNE_CKPT`"`n")
  }
  if ($text -notmatch 'fine_tune_checkpoint_type') {
    $text = [regex]::Replace($text, '(fine_tune_checkpoint:.*\r?\n)', '$&  fine_tune_checkpoint_type: "detection"`n')
  }
  Write-Host "Set fine_tune_checkpoint -> $FINE_TUNE_CKPT"
} else {
  Write-Warning "Fine-tune checkpoint not found at $FINE_TUNE_CKPT. Skipping fine_tune_checkpoint edit."
}

$patternTrain = '(?s)train_input_reader:\s*\{.*?\}'
$replacementTrain = "train_input_reader: {`n  label_map_path: `"$LABEL_MAP`"`n  tf_record_input_reader {`n    input_path: `"$TRAIN_RECORD`"`n  }`n}"
if ($text -match $patternTrain) {
  $text = [regex]::Replace($text, $patternTrain, [System.Text.RegularExpressions.Regex]::Replace($replacementTrain, '\\','\\'))
  Write-Host "Replaced train_input_reader block"
} else {
  Write-Warning "train_input_reader block not found. Please edit config manually."
}

$patternEval = '(?s)eval_input_reader:\s*\{.*?\}'
$replacementEval = "eval_input_reader: {`n  label_map_path: `"$LABEL_MAP`"`n  tf_record_input_reader {`n    input_path: `"$VAL_RECORD`"`n  }`n}"
if ($text -match $patternEval) {
  $text = [regex]::Replace($text, $patternEval, $replacementEval, 1)
  Write-Host "Replaced eval_input_reader block (first occurrence)"
} else {
  Write-Warning "eval_input_reader block not found. Please edit config manually."
}

Set-Content -Path $cfgPath -Value $text -Encoding UTF8
Write-Host "Config updated: $cfgPath"

Write-Host "---- Verification (lines containing key fields) ----"
Select-String -Path $cfgPath -Pattern 'num_classes|fine_tune_checkpoint|fine_tune_checkpoint_type|label_map_path|input_path|batch_size|num_steps' |
  Select-Object LineNumber,Line | ForEach-Object { "{0}: {1}" -f $_.LineNumber, $_.Line }
