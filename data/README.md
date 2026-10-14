# AgeWeaver Dataset Protocol & Manifest

## Sourcing & Licensing Principles

Facial age progression research requires strict ethical compliance regarding identity rights, intellectual property, and licensing. In accordance with project requirements:

1. **No Unlicensed Web Scraping**: Arbitrary images retrieved from search engines are not assumed to be in the public domain.
2. **Permitted-Use Foundations**:
   - Primary source: Wikimedia Commons under Creative Commons licenses (CC-BY, CC-BY-SA, or CC0 / Public Domain).
   - Each manifest entry specifies the explicit license type, original author/creator, attribution requirement, and source repository URL.
3. **Privacy and Safety**:
   - Zero inclusion of personal photographs, private individuals, or unverified minors.
   - Celebrity profiles are curated for research benchmarking with documented birth years and known temporal capture metadata where available.
4. **Audit Categories**:
   - `Planned Identities`: Total candidate identities in the manifest (target: >= 100).
   - `Manifest Entries`: Complete documented rows with valid schema attributes.
   - `Acquired Images`: Locally downloaded files matching source URL.
   - `Verified & Usable`: Inspected for single-face detection, clarity, absence of heavy occlusions.
   - `Experimentally Evaluated`: Included in generative age progression benchmarks.

## Manifest Schema (`dataset_manifest.csv`)

| Column Name | Type | Description |
|---|---|---|
| `identity_id` | string | Unique stable identity key (e.g., `ID001_RAJESH_KHANNA`). |
| `display_name` | string | Full name of the subject. |
| `image_id` | string | Unique image asset key (e.g., `IMG_001_01`). |
| `source_url` | string | Canonical web repository URL (e.g. Wikimedia Commons). |
| `source_name` | string | Repository name (e.g., `Wikimedia Commons`). |
| `license` | string | Permitted use license (e.g., `CC BY-SA 3.0`, `CC BY 2.0`, `Public Domain`). |
| `attribution` | string | Required author attribution statement. |
| `birth_year` | integer | Verified birth year of subject. |
| `capture_year` | integer / None | Approximate or verified capture year. |
| `reference_age` | integer / None | Age at capture (`capture_year - birth_year`). |
| `gender` | string | Demographic notation (`male`, `female`, `non-binary`). |
| `quality_status` | string | Audit status (`verified`, `pending_download`, `excluded`). |
| `face_detection_status` | string | InsightFace audit (`detected`, `not_audited`, `no_face`). |
| `inclusion_status` | string | Experiment status (`included`, `candidate`, `excluded`). |
| `notes` | string | Curatorial and provenance observations. |

