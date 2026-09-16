# AgroVoice — Voice-Based Cassava Disease Advisory Agent

Built for the Sahara CodeSwitch Africa Challenge (Agriculture category).

## Problem
Nigeria is the world's largest cassava producer, but diseases like Cassava Mosaic Disease destroy huge portions of smallholder harvests. Most farmers lack access to expert diagnosis and can't use text-based tools easily.

## Solution
A farmer describes a plant problem by voice (in Yoruba-English or Pidgin-English) and uploads a leaf photo. AgroVoice:
1. Transcribes the code-switched speech via Sahara STT
2. Matches described symptoms against a cassava disease knowledge base (built from CABI Plantwise and IITA sources)
3. Classifies the photo using a fine-tuned EfficientNet-B0 model trained on the Cassava Leaf Disease Classification dataset
4. Fuses both signals into a ranked, confidence-scored diagnosis
5. Generates a concrete action plan (treatment or resistant-planting-material guidance)
6. Reads the response back via Sahara TTS, in the farmer's own language

## Architecture
- `backend.py` — FastAPI server handling STT, CV inference, symptom matching, fusion, and TTS
- `AgroVoice.html` — frontend web interface
- CV training notebook — EfficientNet-B0 fine-tuning on the Cassava Leaf Disease dataset

## Benchmark
See the benchmark report (`.tex`) for a full comparison of 4 speech models (Sahara, Whisper base, Whisper-small-Yoruba, Wav2Vec2-XLSR-53 Nigerian Pidgin) across 17 code-switched clips, including a fairness/disparity analysis, plus a TTS intelligibility benchmark.

## Status
Submitted for the Sahara CodeSwitch Africa Challenge, September 2026. Documentation and code cleanup ongoing post-submission.r methodology and results.
