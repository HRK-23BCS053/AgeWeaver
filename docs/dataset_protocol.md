# AgeWeaver Dataset Acquisition & Ethical Protocol

## 1. Ethical Foundation & Privacy Compliance

Generative facial manipulation involves sensitive biometric and personal data. AgeWeaver adheres to strict scientific and ethical protocols:

1. **Zero Personal Data**: Under no circumstances are personal photographs, private family albums, or photos of non-consenting individuals ingested or stored.
2. **Exclusion of Minors**: No real minors or children are scraped or targeted for synthetic manipulation.
3. **Intellectual Property Respect**: Images are sourced exclusively from public cultural repositories (principally Wikimedia Commons) operating under permissive licenses:
   - Creative Commons Attribution (CC-BY 2.0 / 3.0 / 4.0)
   - Creative Commons Attribution-ShareAlike (CC-BY-SA 2.0 / 3.0 / 4.0)
   - Creative Commons Zero (CC0) / Public Domain
4. **Attribution & Provenance**: Every entry in `data/dataset_manifest.csv` records the author, source URL, and license terms.
5. **Synthetic Identity Benchmark**: For routine testing and CI/CD validation, AgeWeaver utilizes deterministic synthetic portraits generated via diffusion text-to-image with fixed seeds, ensuring 100% privacy compliance and zero copyright entanglement.

## 2. 100-Celebrity Manifest Design

The benchmark manifest (`data/dataset_manifest.csv`) contains 100 globally curated cultural, cinematic, and public figures across varied age brackets, genders, and ethnicities:
- Indian Cinema (Rajesh Khanna, Amitabh Bachchan, Shabana Azmi, Kamal Haasan, Irrfan Khan, Madhuri Dixit, Deepika Padukone, etc.)
- Global Cinema (Tom Hanks, Meryl Streep, Denzel Washington, Anthony Hopkins, Michelle Yeoh, Gong Li, Ken Watanabe, etc.)

### 2.1 The Rajesh Khanna Qualitative Case Study
As requested in project requirements, legendary actor Rajesh Khanna (1942–2012) is designated as a canonical qualitative case study (`ID001_RAJESH_KHANNA`). The manifest links to a verified Bollywood Hungama CC-BY 3.0 archival portrait from Wikimedia Commons, downloaded and verified at `data/images/ID001_RAJESH_KHANNA.jpg` (SHA-256: `9f7ac48006d56021`, detected face confidence: 0.85, estimated baseline age: 55.0 years).

## 3. Five-Tier Dataset Audit Lifecycle

AgeWeaver maintains strict separation between planned data and verified experimental data:

1. **Planned Identities**: 100 unique celebrity identities cataloged with biographical birth dates.
2. **Documented Manifest Entries**: Fully validated rows satisfying all 15 required schema attributes in `dataset_manifest.csv`.
3. **Acquired Images on Disk**: Files physically retrieved into `data/images/` (which is strictly gitignored to prevent committing binary media).
4. **Verified & Usable Faces**: Images verified by InsightFace `buffalo_l` (`det_10g.onnx`) with high detection confidence ($> 0.80$), single face, and no heavy occlusion.
5. **Experimentally Evaluated**: Real identities evaluated across controlled diffusion conditions.

## 4. Ingestion & Verification Workflow

### 4.1 Download Celebrity Portraits
Download an individual identity or batch from the manifest:
```powershell
# Ingest qualitative case study
python scripts/download_dataset.py --identity-id ID001_RAJESH_KHANNA

# Ingest top 10 candidates
python scripts/download_dataset.py --top 10
```
Downloaded files are placed into `data/images/`, and provenance is logged to `data/download_log.json`.

### 4.2 Audit Manifest & Local Disk Storage
Run the automated manifest and face audit validator:
```powershell
python scripts/validate_dataset.py
```
This script audits schema conformance, URL schemes, absence of duplicate identifiers, and validates local directory counts against verified faces.

