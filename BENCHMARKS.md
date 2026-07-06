# Benchmarks & Measurement Plan

## Why This Matters

Molefe's design brief specifies a **dual success metric**:
1. **Backend state changed** — did the task actually complete? (the product metric)
2. **Completed without English** — did the user finish never needing English? (the research metric)

The second metric is where the novelty lives. Nobody has benchmarked this for non-literate voice users. 

The English pipeline is our **control**. The Zulu pipeline is the **treatment**. The gap between them tells us where the cascade is losing performance — and whether the safety layer closes that gap.

---

## Measurement Framework

```
                           ENGLISH BASELINE                    ZULU PIPELINE
                    (control — theoretical ceiling)      (treatment — what we're proving)
                    ─────────────────────────────       ─────────────────────────────
Stage 1 (ASR)       Whisper base                          MMS-1b-all + zul adapter
Stage 2 (MT)        —                                     NLLB-200 distilled 600M
Stage 3 (LLM)       Claude Sonnet 4                       Claude Sonnet 4
Stage 4 (MT)        —                                     NLLB-200 distilled 600M
Stage 5 (TTS)       F5-TTS (English ref)                  F5-TTS (English ref, mismatched)
                    Safety layer: N/A                     Safety layer: on/off
```

---

## A. Per-Stage Technical Benchmarks

### Stage 1: ASR
| Metric | English (Whisper) | Zulu (MMS) | Dataset |
|--------|-------------------|------------|---------|
| **WER** (Word Error Rate) | 4-5% | 20-30% | LibriSpeech / NCHLT-Zulu |
| **CER** (Character Error Rate) | 1-2% | 13-15% | Same |
| **Real-time factor** | 0.1-0.3 | 0.5-1.0 | Locally measured |
| **VAD accuracy** | >95% | >95% | Locally measured |

**Zulu-specific**: track separate WER for (a) single words, (b) short phrases, (c) long conversational utterances. Our anecdotal testing shows accuracy degrades sharply on (c).

### Stage 2: MT
| Metric | zu→en | en→zu | Dataset |
|--------|-------|-------|---------|
| **BLEU** | ~22.7 | ~22.7 | FLORES-200 |
| **COMET** | ~0.75 | ~0.75 | FLORES-200 |
| **chrF++** | — | — | FLORES-200 |
| **Human eval** (1-5) | — | — | Need Zulu speakers |

**Critical for us**: measure BLEU/COMET on morphologically complex Zulu — that's where NLLB is weakest. Standard benchmarks use "clean" formal text; our target user speaks informal conversational Zulu with code-switching.

### Stage 3: LLM (same for both pipelines)
| Metric | Target |
|--------|--------|
| **Task completion accuracy** | >90% (English) |
| **Tool call correctness** | >95% (once tool use is added) |
| **Response relevance** | LLM-as-judge or human eval |
| **Hallucination rate** | <5% |

### Stage 4: MT (reverse)
Same as Stage 2 but critically: measure round-trip fidelity (zu → en → zu vs original zu).

### Stage 5: TTS
| Metric | English (F5-TTS) | Zulu (F5-TTS) |
|--------|------------------|---------------|
| **MOS** (Mean Opinion Score, 1-5) | ~4.0-4.5 | ~2.5-3.0 (mismatched ref) |
| **Intelligibility** (native speakers can transcribe back) | >95% | Unknown — needs testing |
| **Latency** (time to first audio) | ~1-3s | ~1-3s |
| **Naturalness** | High | Low (see issue #1) |

### Safety Layer
| Metric | Target |
|--------|--------|
| **Precision** (fires only when there's a real error) | >80% |
| **Recall** (catches most real errors) | >70% |
| **False positive rate** | <15% (user friction) |
| **False negative rate** | <20% (bad turns proceed) |
| **Latency added** | <2s per triggered turn |

---

## B. End-to-End System Benchmarks

### Dual Success Metric (from the brief)
1. **Backend state changed** — after N turns, did the DB reflect the intended action?
2. **Completion without English** — did the user finish without any English word appearing in their utterances?

**Measurement**: run 20-50 sessions for each task, English pipeline vs Zulu pipeline (with and without safety layer).

### Latency (matters for the infrastructure thesis)
| Metric | English target | Zulu target |
|--------|----------------|-------------|
| **Turn latency** (mic-stop to speaker-start) | <2s | <4s |
| **P95 turn latency** | <5s | <8s |
| **Full task duration** | <60s | <90s |

### User Experience
| Metric | How to measure |
|--------|----------------|
| **SUS score** (System Usability Scale) | Standard 10-question survey after each session |
| **Task-specific UEQ** | User Experience Questionnaire |
| **Trust score** | Would you use this for a real task? (1-7) |
| **Comprehension** | Can user summarize what the agent said? |

---

## C. Research Thesis Benchmarks (the infrastructure-confound)

This is what turns Pillar A from an assertion into a measurement.

### The regression
```
task_completion ~ device + connectivity + power + eng_literacy + native_literacy + ε
```

Measurable variables:
| Variable | How to capture |
|----------|----------------|
| **Device tier** | Phone spec (RAM, chip year) — self-reported |
| **Connectivity** | Measured bandwidth + latency at session start |
| **Power reliability** | Was load-shedding active? Battery % at start/end |
| **English literacy** | Simple pre-test: read 3 English sentences aloud |
| **Native literacy** | Simple pre-test: read 3 Zulu sentences aloud |
| **Outcome: task completion** | Backend-state + no-English metrics |

### The comparison that proves/disproves the thesis
If **native literacy strongly predicts** completion → the field's consensus is right (language IS the barrier).
If **connectivity/power dominate** and native literacy is weak → **our thesis wins** — infrastructure is the real gate.

---

## D. Comparative Benchmarks (English vs Zulu vs Zulu+Safety)

For each of 3-5 realistic tasks (SASSA status check, airtime purchase, etc.), run:

| Condition | Task completion rate | Avg latency | User satisfaction |
|-----------|----------------------|-------------|-------------------|
| **English pipeline** | (ceiling) | (ceiling) | (ceiling) |
| **Zulu pipeline, safety OFF** | ? | ? | ? |
| **Zulu pipeline, safety ON** | ? | ? | ? |

**Success criterion**: safety-on Zulu pipeline reaches ≥75% of English pipeline's completion rate. If yes → the cascade + safety design works for our user.

---

## E. Cross-Language / Multilingual Benchmarks

Where our pipeline components sit vs the state of the art:

### ASR (Zulu)
- **NCHLT-Zulu**: standard South African corpus
- **Common Voice Zulu**: crowdsourced, more informal
- **Compare against**: XLS-R, MMS-1B, W2v-BERT (per arXiv:2512.10968)

### MT (Zulu↔English)
- **FLORES-200**: multilingual benchmark
- **MAFAND-MT**: African languages benchmark
- **Compare against**: NLLB, Google Translate, GPT-4, Claude

### LLM (African languages)
- **AfriMMLU**: multi-choice QA
- **IrokoBench**: broader evaluation
- **MasakhaNER**: named entity recognition

---

## Suggested Order of Implementation

### Phase 1: Session logging (blocks everything)
See issue #3. Nothing is measurable without it.

### Phase 2: Per-stage isolation
Build a test harness that:
- Feeds a known Zulu audio file → captures ASR output → calc WER
- Feeds known Zulu text → captures MT output → calc BLEU
- Feeds known English → captures TTS → calc intelligibility (via ASR round-trip)

### Phase 3: End-to-end benchmarking
- 10 canonical tasks (voice recordings of intended queries)
- Run each through English + Zulu pipelines
- Compute the dual success metric

### Phase 4: User testing
- 5-10 native Zulu speakers
- Real tasks, real recordings, real feedback
- Captures the "completion without English" metric authentically

### Phase 5: Thesis regression
- Once we have 100+ sessions across varied devices/connectivity → run the regression
- This is the paper

---

## Datasets We'll Need

| Dataset | Purpose | Availability |
|---------|---------|--------------|
| **NCHLT-Zulu** | Zulu ASR training/eval | Requires access via SADiLaR |
| **FLORES-200** | MT benchmark | Public (Meta) |
| **Common Voice Zulu** | Additional ASR eval | Public (Mozilla) |
| **MasakhaNER** | NER quality | Public (Masakhane) |
| **AfriMMLU** | LLM eval | Public |
| **Our own recordings** | End-to-end eval | Need to collect |

---

## What We Can Measure RIGHT NOW

With no additional work beyond session logging (#3), we can already start collecting:

1. **Per-turn similarity scores** from the safety layer
2. **Confirm-back trigger rate** (how often does it fire?)
3. **Latency per stage** (already prints in logs, just needs structured capture)
4. **Task success rate** (manual: did each session achieve its goal?)

Everything else needs either:
- Reference datasets (FLORES, NCHLT)
- Human evaluators (native Zulu speakers)
- More sessions (statistical significance)

---

## Publishing Targets

Aligned with Molefe's brief §Appendices:
- **Deep Learning Indaba 2026** — SA venue, mentorship-first
- **AfricaNLP workshop** (ICLR/EMNLP/ACL) — perfect fit for this work
- **WMT African MT shared task** — measurable, ranked
- **Interspeech / ICASSP** — if we produce a TTS paper (see issue #1)
- **EMNLP / ACL main** — the safety-layer paper once we have data

The safety layer + dual success metric IS a paper. It's the unbenchmarked space Molefe called out. We just need the numbers.
