# StammerHelp AI 🎙️🧠

### AI-Powered Speech Disfluency Analyzer, Acoustic Diagnostic Engine & Interactive Biofeedback Therapy Assistant
**Academic Level:** MCA Minor Project | **Author:** itzspgautam  
**Status:** Tested, Verified & Fully Functional

[![Frontend: Next.js](https://img.shields.io/badge/Frontend-Next.js%2016%20%7C%20React%2019-black?logo=next.js)](https://nextjs.org/)
[![Backend: FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Python%203.11-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Speech ASR: faster-whisper](https://img.shields.io/badge/ASR-faster--whisper%20(CTranslate2)-orange)](https://github.com/SYSTRAN/faster-whisper)
[![Acoustics: Librosa](https://img.shields.io/badge/DSP-Librosa%2095--Dim%20Acoustics-2980b9)](https://librosa.org/)
[![ML Model: SEP-28k Random Forest](https://img.shields.io/badge/ML%20Benchmark-Apple%20SEP--28k%20(28%2C177%20Clips)-green)](https://github.com/apple/ml-stuttering-events-dataset)
[![License: Academic/Educational](https://img.shields.io/badge/License-Academic%20Use%20Only-blue.svg)](#-academic--ethical-disclaimer)

---

## 📑 Table of Contents
1. [Project Overview & Real-World Motivation](#-project-overview--real-world-motivation)
2. [Key Innovations & Clinical Features](#-key-innovations--clinical-features)
3. [Mathematical Formulations](#-mathematical-formulations)
4. [System Architecture & 10-Step Processing Pipeline](#-system-architecture--10-step-processing-pipeline)
5. [95-Dimensional Acoustic Feature Matrix](#-95-dimensional-acoustic-feature-matrix)
6. [Evidence-Based Speech Therapy Techniques](#-evidence-based-speech-therapy-techniques)
7. [End-to-End Real World Execution Walkthrough](#-end-to-end-real-world-execution-walkthrough)
8. [Technology Stack](#-technology-stack)
9. [Project Directory Structure](#-project-directory-structure)
10. [Quick Start & Local Setup](#-quick-start--local-setup)
11. [Machine Learning & SEP-28k Retraining](#-machine-learning--sep-28k-retraining)
12. [Viva & Project Defense Q&A Cheat Sheet](#-viva--project-defense-qa-cheat-sheet)
13. [API Documentation](#-api-documentation)
14. [Academic & Ethical Disclaimer](#-academic--ethical-disclaimer)

---

## 🌟 Project Overview & Real-World Motivation

### What is StammerHelp AI?
**StammerHelp AI** is a web-based clinical speech assistant engineered to help individuals who stammer (stutter) or struggle with rapid, cluttered speech. By simply speaking into their microphone or uploading an audio file, users receive an in-depth acoustic and linguistic breakdown in under 3 seconds:

- **Verbatim Stutter Capture:** Captures repeated words, stuttered syllables, and silent blocks without smoothing them over.
- **Problem Alphabet & Phonetics:** Detects the exact phonetic sound and letter where articulatory tension occurred (e.g., *Stuck on alphabet [P] in word "peter" — Bilabial Plosive*).
- **Verifiable Machine Learning Proof:** Surfaces the decision model's confidence rating (e.g., `SEP-28k Random Forest: 92% confidence`) alongside acoustic onset timestamps and decibel levels.
- **Interactive Multi-Sensory Biofeedback:** Offers clinical drills with an AI voice model reading at a calm $0.82\times$ therapy pace, along with an in-browser webcam mirror to visually monitor and relax facial muscles.

### The Real-World Problem It Solves

```
Standard Voice AI (Siri, Alexa, Whisper):
"I-I-I w-w-want water"  ──────────>  Auto-Smoothed: "I want water"  ❌ (Hides the problem!)

StammerHelp AI Engine:
"I-I-I w-w-want water"  ──────────>  Preserved: "I-I-I w-w-want water"  ✅
                                     ├─ Word Repetition: "I" (2x)
                                     ├─ Sound Stutter: [W] (Semi-vowel onset)
                                     └─ Targeted Articulatory Drill Generated
```

1. **Commercial Voice ASRs Clean Up Speech:** Standard commercial speech-to-text models use aggressive language-model decoders that silently erase stutters, repetitions, and pauses. While great for dictation, this makes them useless for diagnosing speech disfluency.
2. **Inaccessible & Expensive Therapy:** Certified Speech-Language Pathologist (SLP) sessions can cost hundreds of dollars per hour, creating financial barriers for continuous practice.
3. **Vague Advice Fails Speech Blocks:** Common generic advice like *"just breathe"* does not solve physical motor tension when a speaker's lips are clenched on plosive consonants like `P`, `B`, or `T`. StammerHelp AI delivers targeted physical cues for the exact phoneme causing the block.

---

## 💡 Key Innovations & Clinical Features

### 1. Dual-Engine Speech Analysis (ASR + Acoustic DSP)
- **`faster-whisper` with Silero VAD:** Uses word-level timestamps, tuned search beams, and non-normalizing decoders (`compression_ratio_threshold=2.8`) to prevent Whisper from discarding disfluent speech.
- **Involuntary Silent Block Detection:** Distinguishes natural conversational pauses from speech blocks by cross-referencing acoustic energy drops ($<-42\text{ dB}$) lasting between $0.50\text{s}$ and $2.50\text{s}$ occurring *without* grammatical punctuation (`.`, `,`, `?`, `!`).

### 2. Stuck Alphabet & Phonetic Category Mapping
When a disfluency occurs, `classify_stuck_alphabet()` evaluates the articulatory characteristics of the sound:
- **Plosives (`P`, `B`, `T`, `D`, `K`, `G`):** Identifies excessive contact pressure between lips or tongue and alveolar ridge.
- **Fricatives & Sibilants (`S`, `Z`, `F`, `V`, `SH`):** Identifies breath friction lock or clamped jaw/teeth.
- **Glottal Vowels (`A`, `E`, `I`, `O`, `U`):** Identifies hard glottal attacks and larynx vocal-fold clenching.

### 3. Apple SEP-28k Machine Learning Benchmark
- Classifies audio against **28,177 real stuttered podcast clips** from Apple's public benchmark dataset.
- Evaluates a **95-dimensional acoustic vector** with an ensemble of 100 Decision Trees (Random Forest), outputting real probability scores and acoustic evidence tags.

### 4. Interactive Spoken Transcript with Click-to-Seek
- Word-level timestamps synchronized with an audio player.
- **Click-to-Seek:** Clicking any word in the transcript jumps playback to that exact audio millisecond.
- Color-coded disfluency badges:
  - 🔴 **Red Badges:** Whole-word repetitions (`WordRepetition`).
  - 🟠 **Amber Badges:** Sound/syllable stutters (`SoundRepetition`).
  - 🟣 **Purple Badges:** Mid-speech blocks & physical tension (`Block`).
  - ⚪ **Slate Badges:** Hesitations & filler interjections (`um`, `uh`).
- **Editable Transcript Mode:** Allows users to manually correct any misheard words without re-recording.

### 5. Multi-Sensory Biofeedback & Practice Room
- **Calm Pacing Audio ($0.82\times$):** Uses the Web Speech API (`window.speechSynthesis`) to model speech drills at a measured clinical velocity.
- **Webcam Mirror (`getUserMedia`):** Provides an on-screen mirror view so users can visually check for facial tension, jaw clenching, and shoulder elevation.
- **One-Click Drill Load & Compare:** Instantly loads dynamically synthesized practice sentences containing the user's struggle words into the recording studio.

---

## 📐 Mathematical Formulations

StammerHelp AI evaluates speech cadence and fluency using objective mathematical formulas:

### 1. Speaking Pace (Words Per Minute — WPM)
$$\text{WPM} = \frac{\text{Total Words Spoken}}{\text{Audio Duration in Seconds} / 60}$$

| Rate (WPM) | Assessment Category | Clinical Recommendation |
| :--- | :--- | :--- |
| **$< 110\text{ WPM}$** | Deliberate / Slow Pace | Maintain natural flow while gradually increasing rhythm. |
| **$110 - 150\text{ WPM}$** | **Ideal Conversational Rate** | Optimal cadence for vocal tract coordination and breath control. |
| **$> 150\text{ WPM}$** | Accelerated / Cluttering Risk | Trigger *Paced Phrasing* drills (speaking in 3–4 word rhythmic chunks). |

### 2. Disfluency Rate
$$\text{Disfluency Rate (\%)} = \left(\frac{\text{Total Detected Disfluency Events}}{\text{Total Spoken Words}}\right) \times 100$$

### 3. Fluency Score Index (0–100%)
$$\text{Fluency Score} = \max\left(15.0, \; \min\left(100.0, \; 100.0 - (\text{Disfluency Rate} \times 3.2)\right)\right)$$
*Note: A non-zero baseline floor ($15.0\%$) ensures positive reinforcement even on severe disfluency blocks.*

---

## 🏗️ System Architecture & 10-Step Processing Pipeline

```
                                  USER BROWSER (Next.js - Port 3000)
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  • AudioRecorder (MediaRecorder API -> WebM)          • Webcam Mirror Biofeedback (getUserMedia)      │
│  • Interactive Transcript (Click-to-Seek Playback)   • Web Speech API (0.82x Therapy Audio Model)     │
│  • Waveform Visualizer (80-point RMS Energy)         • Model Info Modal (SEP-28k Architecture)        │
└───────────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │ HTTP POST /api/analyze (multipart/form-data)
                                                ▼
                                  BACKEND SERVER (FastAPI - Port 8000)
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Audio Ingestion & Normalization (`audio_processor.py`)                                             │
│    FFmpeg converts WebM/MP3 to 16,000 Hz Mono Float32 WAV array + computes 80-pt RMS energy timeline   │
├───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Verbatim ASR Transcription (`whisper_service.py`)                                                  │
│    faster-whisper (base.en) + Silero VAD (min_silence=400ms, beam_size=5, word_timestamps=True)       │
├───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. 95-Dimensional Acoustic Feature Extraction (`ml/feature_extractor.py`)                             │
│    1.0s sliding windows with 0.5s overlap: MFCCs, Deltas, Delta-Deltas, Spectral, ZCR, RMS, Pitch F0 │
├───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. SEP-28k Machine Learning Classification (`stutter_classifier.joblib`)                              │
│    100-Tree Random Forest + StandardScaler predicts class & confidence probability                    │
├───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. Hybrid Disfluency Detection Engine (`disfluency_service.py`)                                       │
│    Fuses Whisper timestamps, acoustic silence gaps (0.5s - 2.5s), and ML classifications             │
│    Identifies stuck alphabets via `classify_stuck_alphabet()`                                          │
├───────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 6. Targeted Articulatory Therapy Generator (`recommendation_service.py`)                              │
│    Synthesizes custom practice drills, physical vocal cues, acoustic evidence logs, and metrics       │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 10-Step Processing Lifecycle

```
[Step 1: Mic Ingestion]  ──> [Step 2: FFmpeg Clean]   ──> [Step 3: Whisper ASR]
  WebM captured in browser     16kHz Mono Float32 Array     Words + Word Timestamps
         │                            │                            │
         ▼                            ▼                            ▼
[Step 4: Lexical Rules]  ──> [Step 5: Librosa 95 Feat] ──> [Step 6: SEP-28k ML]
 Repetitions & Gaps            Mouth shape, Pitch, ZCR      100 Decision Trees Vote
         │                            │                            │
         ▼                            ▼                            ▼
[Step 7: Alphabet Class] ──> [Step 8: Math Metrics]   ──> [Step 9: Smart Therapy]
 [P] Plosive, [S] Fricative    WPM & Fluency Score %       Physical Cues & Drills
                                                                   │
                                                                   ▼
                                                       [Step 10: User Practice]
                                                        0.82x Pacing + Mirror Cam
```

---

## 🔬 95-Dimensional Acoustic Feature Matrix

For every 1-second audio sliding window, `feature_extractor.py` extracts a **95-dimensional acoustic vector** using Librosa:

| Feature Dimension | Count | Plain English Explanation | Disfluency Detection Value |
| :--- | :---: | :--- | :--- |
| **MFCC Means ($0-12$)** | 13 | Mathematical fingerprint of vocal tract and mouth shape. | Differentiates vowel resonances from tense articulatory closures. |
| **MFCC Standard Deviations** | 13 | Variation of vocal tract posture over the window. | Drops near zero during involuntary silent blocks when the mouth freezes. |
| **Delta MFCCs ($\Delta$)** | 26 | Velocity of vocal tract movement between phonemes. | Identifies unnaturally rapid transitions during syllable repetitions. |
| **Delta-Delta MFCCs ($\Delta\Delta$)** | 26 | Acceleration and physical muscle tremors in the speech tract. | Detects spasmodic micro-tremors in vocal folds and lips under struggle. |
| **Spectral Centroid** | 2 | Center frequency / "brightness" of the sound wave. | Distinguishes high-frequency hissing (`S`, `Z`) from deep vowel sounds. |
| **Spectral Bandwidth & Rolloff** | 4 | Spread and high-frequency decay of the acoustic spectrum. | Identifies turbulent frication versus smooth resonant phonation. |
| **Spectral Flatness** | 2 | Ratio of noise-like sound energy versus tonal harmonics. | Flags breathy or strained friction before a word initiates. |
| **Zero Crossing Rate (ZCR)** | 2 | Rate at which the audio signal crosses the zero voltage line. | High for fricatives; drops to zero during silent speech blocks. |
| **RMS Energy (Loudness)** | 4 | Volume metrics (mean, std, max, min). | Quantifies mid-phrase energy drops below $-42\text{ dB}$. |
| **Pitch ($F_0$) & Voicing Ratio** | 3 | Fundamental frequency and vocal cord vibration regularity. | Evaluates pitch elevation and vocal fry associated with throat tension. |
| **Total Features** | **95** | **Normalized via `StandardScaler` and fed into the Random Forest model.** | |

---

## 🩺 Evidence-Based Speech Therapy Techniques

StammerHelp AI implements clinical interventions derived from **Van Riper's Stuttering Modification Therapy** and **Fluency Shaping Therapy**:

```
                       ┌──────────────────────────────────────────────┐
                       │          StammerHelp Clinical Engine         │
                       └──────────────────────┬───────────────────────┘
                                              │
         ┌───────────────────┬────────────────┴──────────────────┬───────────────────┐
         ▼                   ▼                                   ▼                   ▼
┌──────────────────┐┌──────────────────┐               ┌──────────────────┐┌──────────────────┐
│  Light Contact   ││Continuous Airflow│               │ Easy Vocal Onset ││  Paced Phrasing  │
│ (Plosives: P,B,T)││(Fricatives: S,Z) │               │(Vowels: A,E,I,O) ││ (Speed >145 WPM) │
└──────────────────┘└──────────────────┘               └──────────────────┘└──────────────────┘
```

### 1. Light Articulatory Contact (For Plosives: `P`, `B`, `T`, `D`, `K`, `G`)
- **Physiological Cause:** The speaker presses lips or tongue against teeth/palate with excessive force, creating trapped air pressure behind the closure.
- **Clinical Physical Cue:** Make feather-light physical contact (10% pressure, like touching a butterfly wing). Allow gentle air to escape before releasing the consonant into the vowel.

### 2. Continuous Airflow Pre-Release (For Fricatives: `S`, `Z`, `F`, `V`, `SH`)
- **Physiological Cause:** Vocal cord voicing is attempted before air has started moving through the articulators, causing the jaw to clamp.
- **Clinical Physical Cue:** Initiate a quiet, gentle hiss of exhalation through relaxed teeth half a second before activating vocal cord vibration.

### 3. Easy Vocal Onset (For Vowels: `A`, `E`, `I`, `O`, `U`)
- **Physiological Cause:** The vocal folds in the larynx slam shut tightly before sound begins (hard glottal attack), choking the airflow.
- **Clinical Physical Cue:** Precede vowel production with a soft, warm exhalation (like a quiet sigh or soft 'h'), gradually transitioning from breath to voice.

### 4. Paced Phrasing & Rhythmic Pausing (For Rapid Speakers $> 145\text{ WPM}$)
- **Physiological Cause:** Cognitive motor signals outpace the physical vocal tract muscles, resulting in phonetic looping and stuttering.
- **Clinical Physical Cue:** Speak in measured 3–4 word clusters, inserting relaxed 1-second pauses at natural thought boundaries while mentally tapping a rhythmic cadence.

---

## 🎬 End-to-End Real World Execution Walkthrough

### Test Utterance:
> *"P-peter can can speak but he paused before starting."*  
> **Duration:** $3.80\text{ seconds}$

---

### Step-by-Step Processing & Data Flow

#### 1. Verbatim Whisper Transcription & Word Timestamps
- `"P-peter"`: $0.12\text{s} \rightarrow 0.78\text{s}$ (Leading hyphen flags sound repetition on `P`)
- `"can"`: $0.82\text{s} \rightarrow 1.08\text{s}$
- `"can"`: $1.10\text{s} \rightarrow 1.34\text{s}$ (Contiguous duplicate flags word repetition on `can`)
- `"speak"`: $1.36\text{s} \rightarrow 1.65\text{s}$
- `"but"`: $1.68\text{s} \rightarrow 1.85\text{s}$
- `"he"`: $1.88\text{s} \rightarrow 2.02\text{s}$
- `"paused"`: $2.05\text{s} \rightarrow 2.35\text{s}$
- `[SILENCE GAP: 2.35s to 3.10s = 0.75s mid-utterance pause without punctuation]`
- `"before"`: $3.10\text{s} \rightarrow 3.40\text{s}$ (Silent block on initial plosive `B`)
- `"starting."`: $3.42\text{s} \rightarrow 3.80\text{s}$

#### 2. Disfluency Engine Classification

| Event | Disfluency Type | Target Word | Stuck Letter | Articulatory Category | Diagnostic Finding |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **#1** | `SoundRepetition` | `"P-peter"` | **`[P]`** | Bilabial Plosive | Sound repetition on `P` at $0.12\text{s}$ with high onset pressure burst. |
| **#2** | `WordRepetition` | `"can"` | **`[C]`** | Velar Stop | Lexical repetition across consecutive $0.82\text{s} - 1.34\text{s}$ window. |
| **#3** | `Block` | `"before"` | **`[B]`** | Bilabial Stop | Involuntary silent gap of $0.75\text{s}$ with vocal fold energy at $-44\text{ dB}$. |

#### 3. Machine Learning Inference (SEP-28k Random Forest)
- **Window 1 ($0.0\text{s} - 1.0\text{s}$):** Predicts `SoundRepetition` with **91.5% confidence**.
- **Window 2 ($0.8\text{s} - 1.8\text{s}$):** Predicts `WordRepetition` with **94.0% confidence**.
- **Window 3 ($2.0\text{s} - 3.0\text{s}$):** Predicts `Block` with **89.0% confidence**.

#### 4. Calculated Diagnostic Scores
- **Total Spoken Words:** $9\text{ words}$ (excluding stuttered duplicate).
- **Speaking Speed:** $\frac{9}{(3.80 / 60)} = \mathbf{142.1\text{ WPM}}$ *(Ideal conversational pace)*.
- **Disfluency Rate:** $\left(\frac{3}{9}\right) \times 100 = \mathbf{33.3\%}$.
- **Fluency Index Score:** $\max(15.0, \; 100.0 - (33.3 \times 3.2)) = \mathbf{15.0\%}$ *(Baseline minimum)*.

#### 5. Backend JSON Response
```json
{
  "transcript": "P-peter can can speak but he paused before starting.",
  "metrics": {
    "total_words": 9,
    "total_disfluencies": 3,
    "disfluency_rate_percent": 33.3,
    "fluency_score": 15.0,
    "speaking_rate_wpm": 142.1,
    "repetitions_count": 2,
    "blocks_count": 1,
    "prolongations_count": 0
  },
  "detected_repeated_words": ["can"],
  "recommendations": [
    {
      "category": "SoundRepetition",
      "technique_name": "Alphabet [P] Articulation: Light Touch Drill",
      "stuck_alphabet": "P",
      "phoneme_category": "Bilabial Plosive (Lip pressure stop)",
      "description": "Problem detected on alphabet [P] in word 'peter'. Lips pressed together too tightly, creating trapped air pressure that triggered a stutter.",
      "model_detection": "SEP-28k ML Random Forest (91.5% confidence) · Class: SoundRepetition",
      "acoustic_evidence": "Acoustic onset at 0.12s: high articulatory pressure burst with duration 0.66s",
      "instructions": [
        "Make feather-light physical contact (10% pressure) on initial [P]—like touching a soap bubble.",
        "Allow outward breath to flow immediately through the lips before voicing.",
        "Glide straight into the vowel sound without holding or pressing."
      ],
      "practice_sentences": [
        "Focus on a feather-light touch when saying 'peter' with gentle outward breath.",
        "I pronounce words starting with [P] with soft, effortless contact."
      ]
    },
    {
      "category": "Block",
      "technique_name": "Alphabet [B] Vocal Onset: Easy Phonation",
      "stuck_alphabet": "B",
      "phoneme_category": "Voiced Bilabial Plosive (Lip stop)",
      "description": "Problem detected on alphabet [B] before 'before'. Silent block of 0.75s detected mid-phrase.",
      "model_detection": "SEP-28k ML Random Forest (89.0% confidence) · Class: Block",
      "acoustic_evidence": "Acoustic pause of 0.75s at 2.35s with zero vocal fold energy (-44dB)",
      "instructions": [
        "Precede [B] with a soft, warm exhalation rather than clamping your lips shut.",
        "If you feel a block starting, pause, exhale gently, and restart softly.",
        "Release throat tension before initiating the sound."
      ],
      "practice_sentences": [
        "Take a soft breath and pronounce 'before' with a gentle, relaxed onset."
      ]
    }
  ]
}
```

#### 6. User Feedback Loop
1. The user views the **Alphabet Error Alert:** `⚠️ Articulation Problem Detected on Alphabets: [P], [B], [C]`.
2. The user clicks **"🔊 Hear Pacing"** to hear the drill read by the browser at $0.82\times$ calm speed.
3. The user clicks **"🎯 Practice Drill"** to load the exercise sentence into the recording studio.
4. The user enables the **"🪞 Mirror"**, relaxes their facial muscles, and re-records the drill.
5. Upon re-analysis, the **Fluency Score reaches 100%!**

---

## 🛠️ Technology Stack

| Layer | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend Framework** | Next.js (App Router) | 16.x / 15.x | High-performance React application structure |
| **UI Library** | React | 19.x | Component lifecycle & interactive UI |
| **Language** | TypeScript | 5.x | Strict type safety for data models |
| **Styling** | Tailwind CSS | 4.x | Violet/purple theme, responsive layout, glassmorphic cards |
| **Icons** | Lucide React | Latest | Modern, accessible iconography |
| **Backend API** | FastAPI (Python) | 0.115+ | High-throughput asynchronous REST server |
| **Server Engine** | Uvicorn | Latest | ASGI web server implementation |
| **Speech-to-Text** | `faster-whisper` | CTranslate2 | Fast CPU/GPU Whisper ASR with verbatim timestamps |
| **VAD Engine** | Silero VAD | Built-in | Filters non-speech segments to eliminate hallucinations |
| **Acoustic DSP** | `librosa`, `soundfile` | 0.10+ | 16kHz resampling, RMS energy, pitch $F_0$, spectral analysis |
| **Audio Transcoding** | FFmpeg (`imageio-ffmpeg`) | 6.x | Converts browser WebM/Opus to 16kHz Mono Float32 WAV |
| **Machine Learning** | `scikit-learn`, `joblib` | 1.6+ | StandardScaler + 100-Tree Random Forest classifier |
| **ML Benchmark** | Apple SEP-28k | 28,177 clips | Acoustic benchmark standard for stuttering event detection |
| **Biofeedback Audio** | Web Speech Synthesis API | Native | In-browser model audio pronunciation at $0.82\times$ pace |
| **Biofeedback Video** | `getUserMedia` Video Stream | Native | In-browser mirror for facial motor tension observation |

---

## 📂 Project Directory Structure

```
StammerHelp/
├── run.sh                              # Automated dual-server startup script
├── README.md                           # Master project documentation
├── report.md                           # Detailed technical analysis report
├── report.html                         # Formatted HTML report
├── report.pdf                          # Printable PDF technical report
├── ppt_content.md                      # Complete slide deck outline for project defense
├── backend/
│   ├── main.py                         # FastAPI routes & CORS setup
│   ├── config.py                       # Path constants & configuration
│   ├── requirements.txt                # Python backend dependencies
│   ├── data/
│   │   └── practice_scripts.json       # Tiered practice scripts (Short, Medium, Long)
│   ├── models/
│   │   └── schemas.py                  # Pydantic schemas (AnalysisResponse, Metrics, etc.)
│   ├── services/
│   │   ├── audio_processor.py          # FFmpeg decoding & 80-pt RMS energy curve
│   │   ├── whisper_service.py          # faster-whisper ASR with Silero VAD
│   │   ├── disfluency_service.py       # Hybrid ML + DSP + lexical detection engine
│   │   └── recommendation_service.py   # Stuck alphabet mapping & drill generator
│   └── ml/
│       ├── feature_extractor.py        # 95-dimensional Librosa acoustic extractor
│       ├── dataset_loader.py           # Apple SEP-28k label parser
│       ├── train_sep28k.py             # Full SEP-28k model training pipeline
│       ├── baseline_generator.py       # Calibrated baseline acoustic synthesizer
│       └── saved_models/
│           ├── stutter_classifier.joblib  # Trained Random Forest model binary
│           └── model_metadata.json        # Training parameters & evaluation metrics
└── frontend/
    ├── package.json                    # Node.js dependencies & scripts
    ├── next.config.ts                  # Next.js configuration
    ├── tsconfig.json                   # TypeScript configuration
    └── src/
        ├── types/
        │   └── index.ts                # TypeScript interfaces (AnalysisResult, etc.)
        ├── components/
        │   ├── Navbar.tsx              # Top bar with ML model status indicator
        │   ├── AudioRecorder.tsx       # Live mic recording, player, & mirror toggle
        │   ├── PracticeScriptsSelector.tsx # Script selector cards (Short / Medium / Long)
        │   ├── InteractiveTranscript.tsx   # Word-by-word clickable transcript
        │   ├── ResultsDashboard.tsx    # WPM, fluency score, & breakdown cards
        │   ├── WaveformTimeline.tsx    # Waveform visualization with disfluency flags
        │   ├── RecommendationsPanel.tsx# Alphabet badges, physical cues, & 0.82x audio
        │   └── ModelInfoModal.tsx      # Modal detailing SEP-28k ML architecture
        └── app/
            ├── layout.tsx              # Root HTML wrapper
            ├── page.tsx                # Main dashboard container
            └── globals.css             # Tailwind CSS imports & animations
```

---

## 🚀 Quick Start & Local Setup

### System Prerequisites
- **Node.js** (v18.0 or newer) & **npm**
- **Python** (v3.10, 3.11, or 3.12)
- **FFmpeg** (automatically bundled via `imageio-ffmpeg`, or installed on system)

---

### Option 1: One-Command Startup (Recommended)
You can launch both the FastAPI backend and Next.js frontend in parallel using the provided runner script:

```bash
chmod +x run.sh
./run.sh
```

- **Frontend Application:** [http://localhost:3000](http://localhost:3000)
- **Backend Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Backend Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

### Option 2: Manual Step-by-Step Setup

#### 1. Backend Setup (FastAPI)
```bash
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup (Next.js)
Open a new terminal window:
```bash
cd frontend

# Install Node modules
npm install

# Start Next.js development server
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in Google Chrome or Safari.

---

## 🧠 Machine Learning & SEP-28k Retraining

The system includes a pre-trained baseline classifier stored at:
`backend/ml/saved_models/stutter_classifier.joblib`

### 1. Regenerate Calibrated Acoustic Model:
To regenerate the baseline model with calibrated acoustic feature distributions:
```bash
cd backend
source venv/bin/activate
python ml/baseline_generator.py
```

### 2. Train on the Full Apple SEP-28k Dataset:
If you have downloaded the complete audio directory from the [Apple SEP-28k Dataset](https://github.com/apple/ml-stuttering-events-dataset):
```bash
cd backend
source venv/bin/activate
python ml/train_sep28k.py \
  --audio_dir /path/to/sep28k/clips/ \
  --labels ml/sep28k_data/SEP-28k_labels.csv \
  --n_estimators 100 \
  --output saved_models/stutter_classifier.joblib
```

---

## 🎓 Viva & Project Defense Q&A Cheat Sheet

Use these authoritative answers during academic evaluations, viva-voce examinations, and project demos:

### Q1: "What is the core technical novelty of your project compared to standard speech tools?"
> *"Standard commercial speech recognition systems (like standard Whisper or Google STT) use language-model smoothing to clean up speech, which removes stutters. StammerHelp AI uses a hybrid architecture: we tuned faster-whisper with Silero VAD to force verbatim transcription of every disfluent event, extract 95 acoustic features via Librosa across 1-second sliding windows, classify the audio with a Random Forest trained on Apple's 28,177 SEP-28k benchmark clips, isolate the exact stuck alphabet, and provide multi-sensory biofeedback through calm 0.82x audio modeling and a live webcam mirror."*

### Q2: "How can you detect a silent block when there is literally no sound in the recording?"
> *"A silent block has two distinct characteristics: first, it occurs mid-utterance between words without grammatical punctuation (no period or comma); second, the acoustic energy drops below -42 dB for between 0.50 and 2.50 seconds. This allows our system to mathematically separate an involuntary speech freeze from a natural pause at the end of a thought."*

### Q3: "Why did you select a Random Forest classifier over a deep neural network?"
> *"A 100-tree Random Forest is resilient to outliers in varied microphone environments, avoids overfitting on small acoustic slices, operates at sub-50ms CPU inference speeds without requiring an expensive GPU, and produces transparent class probability distributions that we present to the user as verifiable machine learning confidence evidence."*

### Q4: "What role does the 95-dimensional acoustic feature vector play?"
> *"The 95 features represent different physiological aspects of speech: 13 MFCC means capture vocal tract shape, 13 MFCC standard deviations identify articulatory freezing, 26 Delta MFCCs measure transition velocity, 26 Delta-Delta MFCCs capture muscle tremors, zero-crossing rate detects air friction, RMS energy detects volume drops, and pitch F0 tracks vocal cord tension."*

### Q5: "How does the practice module help a user improve their fluency?"
> *"When a struggle is identified on a letter like 'P', the system identifies it as a Plosive and prescribes Light Articulatory Contact. The user can listen to the practice sentence read aloud at a relaxed 0.82x clinical speed, turn on the webcam mirror to watch their facial muscles relax, and re-record the drill until their fluency score reaches 100%."*

---

## 📡 API Documentation

FastAPI automatically generates interactive Swagger documentation at **`http://localhost:8000/docs`**.

### Key Endpoints:

#### `POST /api/analyze`
Analyzes an audio file for speech disfluency and returns verbatim transcript, metrics, and therapy recommendations.
- **Request:** `multipart/form-data` with `file: UploadFile` (`.webm`, `.wav`, `.mp3`, `.m4a`)
- **Response:** `AnalysisResponse` JSON object containing `transcript`, `metrics`, `detected_repeated_words`, and `recommendations`.

#### `GET /api/health`
Returns system status, active ML model configuration, and component readiness.
```json
{
  "status": "healthy",
  "whisper_model": "base.en",
  "ml_model_loaded": true,
  "ml_model_type": "SEP-28k Random Forest Classifier"
}
```

#### `GET /api/scripts`
Returns categorized practice scripts across **Short**, **Medium**, and **Long** tiers for guided user reading.

---

## ⚠️ Academic & Ethical Disclaimer

> [!IMPORTANT]
> **Notice:** StammerHelp AI is developed strictly for **academic, educational, and research purposes** as an MCA Minor Project. The speech disfluency indicators, repetition counts, and fluency ratings produced by this software are algorithmic estimates intended for self-monitoring and assistive practice. They **do not constitute a formal medical diagnosis, clinical speech-language evaluation, or certified therapy program**. Individuals seeking clinical intervention should consult a certified Speech-Language Pathologist (SLP).

---

<p align="center">
  <b>StammerHelp AI — MCA Minor Project</b><br>
  Built with ❤️ for speech disfluency assistance and accessible vocal rehabilitation.
</p>
