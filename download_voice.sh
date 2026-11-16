#!/usr/bin/env bash

mkdir -p ./voices

voices=(
  "bn_BD-google-medium"
  "en_US-amy-medium"
  "en_US-hifi_tts_92-medium"
  "en_US-lessac-medium"
  "en_US-ljspeech-high"
  "en_US-mike-medium"
  "en_US-sam-medium"
)

for voice in "${voices[@]}"; do
  echo "Downloading: $voice"
  python -m piper.download_voices "$voice" --data-dir ./voices
done
