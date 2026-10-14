"""
CLI tool for downloading and verifying celebrity benchmark portraits from dataset_manifest.csv.
Respects rate limits, checks image validity, audits face detection, and logs provenance.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from ageweaver.dataset import DatasetManager
from ageweaver.utils import setup_logger

logger = setup_logger("download_dataset")

def main():
    parser = argparse.ArgumentParser(description="AgeWeaver Dataset Ingestion & Verification CLI")
    parser.add_argument("--identity-id", type=str, default=None, help="Specific identity_id to download (e.g. ID001_RAJESH_KHANNA)")
    parser.add_argument("--top", type=int, default=None, help="Download top N candidates from manifest")
    parser.add_argument("--manifest", type=str, default="data/dataset_manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--images-dir", type=str, default="data/images", help="Target directory for downloaded images")
    parser.add_argument("--overwrite", action="store_true", help="Force re-download even if file exists")

    args = parser.parse_args()

    dm = DatasetManager(manifest_path=args.manifest, images_dir=args.images_dir)
    manifest = dm.load_manifest()
    print(f"Loaded manifest with {len(manifest)} identities from {args.manifest}")

    if args.identity_id:
        target_ids = [args.identity_id]
    elif args.top:
        target_ids = list(manifest.keys())[:args.top]
    else:
        print("Please specify --identity-id <ID> (e.g. --identity-id ID001_RAJESH_KHANNA) or --top <N>")
        sys.exit(1)

    print(f"Beginning ingestion for {len(target_ids)} target identities...")
    success_count = 0
    fail_count = 0

    for idx, iid in enumerate(target_ids, start=1):
        try:
            print(f"\n[{idx}/{len(target_ids)}] Ingesting {iid} ({manifest[iid].display_name})...")
            path, meta = dm.download_image(iid, overwrite=args.overwrite)
            print(f"  [SUCCESS] Saved to: {path}")
            print(f"  License: {meta.get('license')} | Attribution: {meta.get('attribution')}")
            print(f"  SHA-256: {meta.get('sha256')}")
            print(f"  Face Detected: {meta.get('face_detected')} | Est. Age: {meta.get('estimated_age')}")
            success_count += 1
        except Exception as e:
            print(f"  [FAIL] Ingestion failed for {iid}: {e}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(" INGESTION SUMMARY")
    print(f" Total Requested: {len(target_ids)} | Succeeded: {success_count} | Failed: {fail_count}")
    print(f" Provenance log updated at: {dm.log_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()

