# NOTE: Continue (not Stop) — Windows PowerShell 5.1 turns harmless native
# stderr warnings (e.g. "seed set to 0") into terminating errors under Stop.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

# re-apply the numpy monotonic-align patch (a pip reinstall removes it)
Copy-Item numpy_monotonic_align.py venv\Lib\site-packages\piper\train\vits\monotonic_align\__init__.py -Force

# auto-resume from the newest checkpoint
$ckpt = Get-ChildItem -Recurse -Path lightning_logs -Filter *.ckpt -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$ckptArgs = @()
if ($ckpt) {
    $ckptArgs = @("--ckpt_path=$($ckpt.FullName)")
    Write-Host "Resuming from $($ckpt.FullName)" -ForegroundColor Cyan
}

.\venv\Scripts\python.exe -u -m piper.train fit `
  --data.voice_name mn_voice `
  --data.csv_path piper_data\metadata.csv `
  --data.audio_dir C:\Users\hi\Documents\chatterbox-finetuning\MyTTSDataset_combined\wavs `
  --data.config_path piper_data\voice_config.json `
  --data.phonemes_path piper_data\phonemes.json `
  --data.cache_dir piper_data\cache `
  --data.espeak_voice en-us `
  --data.phoneme_type text `
  --data.batch_size 8 `
  --data.num_workers 2 `
  --data.validation_split 0.02 `
  --model.sample_rate 16000 `
  --model.filter_length 1024 `
  --model.hop_length 256 `
  --model.win_length 1024 `
  --model.mel_channels 80 `
  --model.mel_fmin 0.0 `
  --model.mel_fmax 8000.0 `
  --trainer.devices 1 `
  --trainer.max_epochs 10000 `
  --trainer.check_val_every_n_epoch 2 `
  --trainer.callbacks ModelCheckpoint `
  --trainer.callbacks.every_n_train_steps 600 `
  --trainer.callbacks.save_last true `
  --trainer.callbacks.save_on_train_epoch_end false `
  --trainer.enable_progress_bar false `
  @ckptArgs 2>&1 | Tee-Object -FilePath training.log
