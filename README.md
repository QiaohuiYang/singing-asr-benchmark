# Singing ASR Benchmark

> A small-scale study of ASR robustness under the speech-to-singing domain shift.

## Overview

Automatic speech recognition (ASR) systems have achieved strong performance on conversational speech, but singing introduces substantially different acoustic characteristics, including sustained vowels, pitch variation, rhythmic changes, and longer phonetic durations.

This project investigates whether a pretrained ASR foundation model remains robust when the same utterances are spoken versus sung.

I use **Whisper** as a fixed pretrained ASR baseline and compare recognition performance on paired speech and singing recordings in **English and French**.

---

## Research Question

### Main Question

**How robust is a pretrained ASR foundation model to the domain shift from speech to singing?**

### Secondary Questions

1. Does singing degrade ASR performance relative to speech?
2. What types of recognition errors increase under singing?
3. Are singing-related errors associated with language, utterance length, or acoustic characteristics?

---

## Experimental Design

The core idea is a **paired comparison**.

Each sentence is recorded twice:

- **Speech** — natural spoken reading
- **Singing** — the same sentence sung using a simple melody

This paired design controls for linguistic content and allows the analysis to focus on the effect of singing.

### Dataset

| Property | Design |
|---|---|
| Languages | English, French |
| Utterance groups | Short, Medium, Long |
| Utterances per language | 9 |
| Paired utterances | 18 |
| Total recordings | 36 |
| Conditions | Speech, Singing |

The English and French sentences are semantically corresponding translations. Because the two languages have different phonological and syllabic structures, the melodies are adapted naturally rather than enforcing exact note-to-syllable alignment.

### Singing Setup

To reduce unnecessary variation, the singing recordings use a simple melodic structure:

- C major
- Mostly C4–G4
- Approximately 90 BPM
- 4/4 meter
- Limited vibrato and ornamentation
- Natural singing style

---

# Method

## ASR Model

The experiment uses **Whisper**, a pretrained multilingual speech recognition foundation model.

The model is used **without fine-tuning**. This allows the experiment to measure how a fixed pretrained ASR system behaves under a speech-to-singing domain shift.

Inference is performed using [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper).

## Evaluation Metrics

Recognition performance is evaluated using:

### Word Error Rate (WER)

\[
WER = \frac{S + D + I}{N}
\]

where:

- \(S\) = substitutions
- \(D\) = deletions
- \(I\) = insertions
- \(N\) = number of words in the reference

### Character Error Rate (CER)

CER provides a character-level measure of recognition accuracy and complements WER.

### Error Decomposition

Recognition errors are additionally decomposed into:

- Substitutions
- Deletions
- Insertions

Because the experiment uses paired recordings, speech and singing WER/CER are compared using the **Wilcoxon signed-rank test**.

---

# Results

## 1. Singing substantially degrades ASR performance

Across **18 paired utterances**:

| Metric | Speech | Singing | Mean Δ |
|---|---:|---:|---:|
| WER | 0.134 | 0.408 | **+0.274** |
| CER | 0.064 | 0.177 | **+0.112** |

The paired Wilcoxon signed-rank tests give:

| Metric | p-value | Effect size \(r\) |
|---|---:|---:|
| WER | **0.0068** | **0.781** |
| CER | **0.0192** | **0.649** |

These results provide evidence that singing substantially reduces recognition accuracy relative to speech in this dataset.

### Paired WER

![Paired WER](results/figures/fig1_paired_wer.png)

Each row represents one paired utterance. The figure highlights the change in WER from speech to singing.

The magnitude of degradation varies considerably across utterances: some remain relatively robust, while others show severe increases in WER.

---

## 2. Singing-related errors are primarily driven by substitutions

After normalizing error counts by reference word count:

| Error Type | Speech | Singing |
|---|---:|---:|
| Substitution | 0.115 | **0.296** |
| Deletion | 0.005 | 0.057 |
| Insertion | 0.014 | 0.055 |

The largest increase occurs in **substitution errors**.

The increase in substitutions is substantially larger than the increases in deletion and insertion rates.

This suggests that singing does not simply cause the ASR system to miss portions of the signal. Instead, the model frequently maps sung acoustic patterns to **incorrect lexical hypotheses**.

### Normalized Error Types

![Error Types](results/figures/fig4_error_types.png)

---

## 3. The effect varies across languages

![WER by Language](results/figures/fig2_wer_language.png)

In this dataset, French shows higher WER than English in both speech and singing conditions.

The English and French datasets contain the same number of paired utterances, but the current sample is too small to make a general claim about language difficulty.

Therefore, the language comparison is treated as an **exploratory analysis**.

---

## 4. Sentence length does not show a simple monotonic effect

![ΔWER by Length](results/figures/fig3_delta_wer_length.png)

The increase in WER does not follow a simple pattern in which longer utterances always produce greater singing-related degradation.

The medium-length condition shows relatively small degradation compared with the short and long conditions.

Because each length group contains only six paired observations, this analysis is exploratory.

---

# Qualitative Error Analysis

Quantitative metrics show that singing increases ASR error, but individual transcription examples provide additional insight into how the model fails.

## English Examples

### 1. Local lexical substitution

**Reference**

> The morning feels warm and bright.

**Speech prediction**

> The morning feels warm and bright.

**Singing prediction**

> The morning feels woman bright.

The singing condition introduces a localized lexical substitution:

`warm → woman`

Most of the utterance remains correct, but a single lexical substitution increases WER.

---

### 2. Severe phrase-level distortion

**Reference**

> A quiet melody can make an ordinary evening feel special.

**Speech prediction**

> A quiet melody can make an ordinary evening feel special.

**Singing prediction**

> "Ah, quiet melody can't make it Hold on to the rain evening few special"

The speech recording is transcribed correctly, whereas the singing recording exhibits substantial phrase-level distortion.

Interestingly, the output still consists largely of plausible English words rather than completely unintelligible noise. This suggests that the model may map the sung acoustic signal to an incorrect but locally plausible lexical sequence.

---

### 3. Singing does not always cause failure

**Reference**

> Learning a new language takes time and patience.

**Speech prediction**

> Learning a new language takes time and patience.

**Singing prediction**

> Learning a new language takes time and patience.

Both conditions are transcribed correctly.

This demonstrates that singing-related degradation is **heterogeneous rather than deterministic**.

---

## French Examples

### 1. Local lexical and phonetic distortion

**Reference**

> Le matin est doux et lumineux.

**Speech prediction**

> Le matin est tout et lumineux.

**Singing prediction**

> Le matin est tout élimineux.

The spoken version contains a localized lexical substitution, while the singing version introduces additional phonetic and lexical distortion.

---

### 2. Severe phrase-level distortion

**Reference**

> Une mélodie calme peut rendre une soirée ordinaire spéciale.

**Speech prediction**

> Une mélodie galme peut rendre une soirée audinaire spéciale.

**Singing prediction**

> Une mèle d'hégalme beurre en haine soirée au dix dix spéciales

The singing condition produces substantial phrase-level distortion, while the speech condition contains only localized errors.

This provides a qualitatively similar failure pattern to the severe English example, suggesting that singing-related recognition errors are not restricted to a single language.

---

### 3. Singing can occasionally outperform speech

**Reference**

> Parfois, j'écoute de la musique avant de dormir.

**Speech prediction**

> Parfois, j'ai goutte la musique avant d'autant remir.

**Singing prediction**

> Parfois je goutte de la musique avant de dormir.

In this case, the singing transcription is actually closer to the reference than the speech transcription.

This provides an important counterexample to the overall degradation trend and reinforces the observation that singing-related ASR degradation varies substantially across utterances.

---

## Summary of Qualitative Patterns

The examples above illustrate four recurring patterns:

1. **Localized substitutions** — a small number of words are replaced by acoustically similar alternatives.
2. **Phrase-level distortion** — singing can produce much larger deviations from the reference.
3. **Cross-language consistency** — similar failure patterns appear in both English and French.
4. **Heterogeneous effects** — some sung utterances remain correct, and occasional cases even outperform speech.

---

#