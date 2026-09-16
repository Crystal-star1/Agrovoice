# AgroVoice Code-Switching Benchmark Audio Dataset

Audio samples used to benchmark speech recognition models for AgroVoice, a voice-based cassava disease advisory agent built for the Sahara CodeSwitch Africa Challenge (Agriculture category).

## Description
Volunteers simulated a Nigerian smallholder farmer describing cassava leaf disease symptoms to a voice-based advisory assistant. Recordings are spontaneous descriptions from a scenario prompt, not scripted reading, to preserve natural code-switching.

## Consent
All recordings were made by consenting volunteers who were informed their voice would be used for this hackathon submission. No personally identifying information is included in the recordings or this dataset.

## Metadata

| File | Language Pair | Domain | Approx. Duration | Device Type | Noise Condition |
|---|---|---|---|---|---|
| Data_1... | Pidgin-English | Agriculture (cassava disease) | ~20–30s | Smartphone voice memo | Quiet indoor |
| Data_2... | Pidgin-English | Agriculture (cassava disease) | ~20–30s | Smartphone voice memo | Quiet indoor |
| Data_3... | Yoruba-English | Agriculture (cassava disease) | ~20–30s | Smartphone voice memo | Quiet indoor |
| ... | ... | ... | ... | ... | ... |

*(Fill in actual duration/device/noise per file — even a rough estimate per clip is enough to satisfy the challenge's stated metadata fields: language pair, domain, accent/country, device type, noise conditions.)*

**Accent/Country:** All speakers are Nigerian (Yoruba-English and Nigerian Pidgin-English speakers).

## Format
Original files are `.m4a`/`.ogg`; converted to 16kHz mono `.wav` for benchmarking.

## Usage
Used to benchmark four ASR models (Sahara, Whisper base, Whisper-small-Yoruba, Wav2Vec2-XLSR-53 Nigerian Pidgin) — see the full benchmark report for methodology and results.
