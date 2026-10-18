# AgeWeaver Dataset Acquisition & Ethical Protocol

## 1. Ethical Foundation & Privacy Compliance

Generative facial manipulation involves sensitive biometric and personal data. AgeWeaver adheres to strict scientific and ethical protocols:

1. **Zero Personal Data**: Under no circumstances are personal photographs, private family albums, or photos of non-consenting individuals ingested or stored.
2. **Exclusion of Minors**: No real minors or children are scraped or targeted for synthetic manipulation.
3. **Intellectual Property Respect**: Images are sourced exclusively from public cultural repositories (principally Wikimedia Commons) operating under permissive licenses:
   - Government Open Data License - India (GODL-India)
   - Creative Commons Attribution (CC-BY 2.0 / 3.0 / 4.0)
   - Creative Commons Attribution-ShareAlike (CC-BY-SA 2.0 / 3.0 / 4.0)
   - Creative Commons Zero (CC0) / Public Domain
4. **Attribution & Provenance**: Every entry in `data/dataset_manifest.csv` records the author, source URL, license terms, capture year, and reference age.
5. **Synthetic Identity Benchmark**: For routine testing and CI/CD validation, AgeWeaver provides deterministic synthetic portraits generated via diffusion text-to-image with fixed seeds, ensuring 100% privacy compliance and zero copyright entanglement.

## 2. 100-Celebrity Manifest Design & Young-to-Old Experimental Design

The benchmark manifest (`data/dataset_manifest.csv`) contains 100 globally curated cultural, cinematic, and public figures across varied age brackets, genders, and ethnicities.

### 2.1 The Young Reference Mandate
A core scientific requirement of facial age-progression research is that **the source image must depict the subject during their younger years (approx. 20–35 years old)**, and the diffusion model must progress facial morphology forward to older target ages ($y_{\text{tgt}} > y_{\text{src}}$, e.g. 45, 60, 70).

Using an older reference portrait (e.g. age 55+) and targeting age 45 fundamentally invalidates the experimental design by transforming it into an age-rejuvenation task rather than age progression.

### 2.2 Verified Younger Reference Identities

#### Identity 1: Rajesh Khanna (`ID001_RAJESH_KHANNA`)
- **Historical Subject**: Born 29 December 1942.
- **Reference Image**: Historic 1970 superstar publicity portrait (from the era of *Aradhana*, *Kati Patang*, *Sachaa Jhutha*), reproduced in the 2013 India Post commemorative stamp issue *Theatre & Cinema (Films & Film Stars)*.
- **Source**: Wikimedia Commons (`File:Rajesh_Khanna_2013_stamp_of_India_(cropped).jpg`).
- **Direct URL**: `https://upload.wikimedia.org/wikipedia/commons/a/a1/Rajesh_Khanna_2013_stamp_of_India_%28cropped%29.jpg`
- **Capture Year**: ~1970 (Documented Chronological Age: **~28 years old**).
- **License**: Government Open Data License - India (GODL-India).
- **Attribution**: India Post, Government of India.
- **Image Dimensions**: 302 x 403 pixels.
- **SHA-256**: `702bd522419f1ccd`
- **Biometric Face Audit**: InsightFace `buffalo_l` single face detected, Bbox `[59.2, 118.3, 241.2, 328.5]`, detection score: `0.785`.
- **Note on Age Estimation**: InsightFace `genderage.onnx` yields an automated estimate of ~41.0 years due to stamp engraving texture and halftone stippling. In accordance with zero-fabrication research standards, AgeWeaver explicitly records both the documented historical chronological age (28 years) and the automated CNN estimate (41.0 years), never conflating the two.

#### Identity 2: Deepika Padukone (`ID009_DEEPIKA_PADUKONE`)
- **Historical Subject**: Born 5 January 1986.
- **Reference Image**: Young adult portrait crop from December 2008 (*Om Shanti Om* / *Bachna Ae Haseeno* era).
- **Source**: Wikimedia Commons (`File:Deepika_Padukone_(face).jpg`).
- **Direct URL**: `https://upload.wikimedia.org/wikipedia/commons/8/8a/Deepika_Padukone_%28face%29.jpg`
- **Capture Year**: 2008 (Documented Chronological Age: **~22 years old**).
- **License**: CC BY-SA 3.0.
- **Attribution**: Bollywood Hungama / Wikimedia Commons.
- **Image Dimensions**: 200 x 200 pixels.
- **SHA-256**: `d6f644e66d13d335`
- **Biometric Face Audit**: InsightFace `buffalo_l` single face detected, Bbox `[58.5, 33.3, 142.6, 153.9]`, detection score: `0.808`, CNN estimated age: 28.0 years.

## 3. Five-Tier Dataset Audit Lifecycle

AgeWeaver maintains strict separation between planned data and verified experimental data:

1. **Manifest Entries (Cataloged)**: 100 unique celebrity identities cataloged with biographical birth dates.
2. **Images Actually Downloaded**: Files physically retrieved into `data/images/` (strictly gitignored to prevent committing binary media).
3. **Images Successfully Validated**: Readable format, uncorrupted headers, single face detected by InsightFace.
4. **Verified Young References**: Documented provenance, verified young chronological age (20–35 years), suitable for genuine forward age progression.
5. **Experimentally Completed**: Real identities evaluated across controlled diffusion conditions with full logs in `outputs/experiments/experiment_log.jsonl`.

## 4. Ingestion & Audit Workflow

### 4.1 Download Celebrity Portraits
Download an individual identity or batch from the manifest:
```powershell
# Ingest verified young Rajesh Khanna reference
python scripts/download_dataset.py --identity-id ID001_RAJESH_KHANNA --overwrite

# Ingest verified young Deepika Padukone reference
python scripts/download_dataset.py --identity-id ID009_DEEPIKA_PADUKONE --overwrite
```
Downloaded files are placed into `data/images/`, and provenance is logged to `data/download_log.json`.

### 4.2 Comprehensive Dataset Audit
Run the dataset audit tool to produce real-time machine-readable summaries:
```powershell
python scripts/audit_dataset.py
```
This tool inspects every entry, checks local disk availability, confirms young-reference criteria, and generates:
- `data/dataset_audit.json`
- `data/dataset_audit.md`
