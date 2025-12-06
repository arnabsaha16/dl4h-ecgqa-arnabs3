import argparse
import pandas as pd
import os, shutil, hashlib

def sample_ecg_data(df, label_col, proportion, dataset_root, output_root,
                    dataset_type="ptbxl", random_state=42):
    """
    Sample ECG records from PTB-XL or MIMIC-IV-ECG and regenerate metadata + waveform files.

    Args:
        df (pd.DataFrame): Metadata DataFrame (ptbxl_database.csv or record_list.csv).
        label_col (str): Column name with class labels (must exist in df).
        proportion (float): Fraction of dataset to sample (e.g., 0.1 for 10%).
        dataset_root (str): Path to original dataset root.
        output_root (str): Path to save sampled subset.
        dataset_type (str): "ptbxl" or "mimic".
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Sampled DataFrame of records.
    """
    os.makedirs(output_root, exist_ok=True)

    if dataset_type == "ptbxl":
        # Stratified sampling at record level
        sampled_df = (
            df.groupby(label_col, group_keys=False)
              .apply(lambda x: x.sample(frac=proportion, random_state=random_state))
              .reset_index(drop=True)
        )

        # Copy both LR and HR waveform files
        for _, row in sampled_df.iterrows():
            for fname in [row["filename_lr"], row["filename_hr"]]:
                for ext in [".dat", ".hea"]:  # enforce .dat before .hea
                    src = os.path.join(dataset_root, fname + ext)
                    dst = os.path.join(output_root, fname + ext)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if os.path.exists(src):
                        shutil.copy(src, dst)

        regenerate_ptbxl_metadata(sampled_df, dataset_root, output_root)

    elif dataset_type == "mimic":
        # Sample subjects, then include all their studies
        sampled_subjects = (
            df["subject_id"].drop_duplicates()
              .sample(frac=proportion, random_state=random_state)
        )
        sampled_df = df[df["subject_id"].isin(sampled_subjects)]

        # Copy waveform files preserving nested structure
        for _, row in sampled_df.iterrows():
            rel_path = row["path"]  # e.g. p1000/p10001725/s41420867/41420867
            for ext in [".dat", ".hea"]:  # enforce .dat before .hea
                src = os.path.join(dataset_root, rel_path + ext)
                dst = os.path.join(output_root, rel_path + ext)
                os.makedirs(os.path.dirname(dst), exist_ok=True)  # create nested folders
                
                if os.path.exists(src):
                    shutil.copy(src, dst)

        regenerate_mimic_metadata(sampled_df, dataset_root, output_root)

    return sampled_df


def regenerate_ptbxl_metadata(sampled_df, dataset_root, output_root):
    """Regenerate PTB-XL metadata files for sampled subset."""
    sampled_df.to_csv(os.path.join(output_root, "ptbxl_database.csv"), index=False)

    translated_path = os.path.join(dataset_root, "ptbxl_database_translated.csv")
    if os.path.exists(translated_path):
        translated = pd.read_csv(translated_path)
        translated_subset = translated[translated["filename_lr"].isin(sampled_df["filename_lr"])]
        translated_subset.to_csv(os.path.join(output_root, "ptbxl_database_translated.csv"), index=False)

    # RECORDS file: all LR first, then all HR
    records_path = os.path.join(output_root, "RECORDS")
    with open(records_path, "w") as f:
        for fname in sampled_df["filename_lr"]:
            f.write(fname + "\n")
        for fname in sampled_df["filename_hr"]:
            f.write(fname + "\n")

    for fname in ["scp_statements.csv", "LICENSE", "example_physionet.py"]:
        src = os.path.join(dataset_root, fname)
        if os.path.exists(src):
            shutil.copy(src, output_root)

    sha_path = os.path.join(output_root, "SHA256SUMS")
    with open(sha_path, "w") as f:
        for _, row in sampled_df.iterrows():
            for fname in [row["filename_lr"], row["filename_hr"]]:
                for ext in [".dat", ".hea"]:  # enforce .dat before .hea
                    file_path = os.path.join(output_root, fname + ext)
                    if os.path.exists(file_path):
                        sha = hashlib.sha256(open(file_path, "rb").read()).hexdigest()
                        rel_path = os.path.relpath(file_path, output_root)
                        f.write(f"{sha}  {rel_path}\n")


def regenerate_mimic_metadata(sampled_df, dataset_root, output_root):
    """Regenerate MIMIC-IV-ECG metadata files for sampled subset."""
    sampled_df.to_csv(os.path.join(output_root, "record_list.csv"), index=False)

    mm_path = os.path.join(dataset_root, "machine_measurements.csv")
    if os.path.exists(mm_path):
        mm = pd.read_csv(mm_path, low_memory=False)
        # Use study_id if available
        if "study_id" in mm.columns:
            mm_subset = mm[mm["study_id"].isin(sampled_df["study_id"])]
        elif "subject_id" in mm.columns:
            mm_subset = mm[mm["subject_id"].isin(sampled_df["subject_id"])]
        else:
            print("Warning: machine_measurements.csv has no study_id/subject_id column, skipping filter.")
            mm_subset = mm
        mm_subset.to_csv(os.path.join(output_root, "machine_measurements.csv"), index=False)

    wn_path = os.path.join(dataset_root, "waveform_note_links.csv")
    if os.path.exists(wn_path):
        wn = pd.read_csv(wn_path, low_memory=False)
        if "study_id" in wn.columns:
            wn_subset = wn[wn["study_id"].isin(sampled_df["study_id"])]
        elif "subject_id" in wn.columns:
            wn_subset = wn[wn["subject_id"].isin(sampled_df["subject_id"])]
        else:
            print("Warning: waveform_note_links.csv has no study_id/subject_id column, skipping filter.")
            wn_subset = wn
        wn_subset.to_csv(os.path.join(output_root, "waveform_note_links.csv"), index=False)

    dict_path = os.path.join(dataset_root, "machine_measurements_data_dictionary.csv")
    if os.path.exists(dict_path):
        shutil.copy(dict_path, output_root)

    # RECORDS file should list file_name values
    #sampled_df["file_name"].to_csv(os.path.join(output_root, "RECORDS"), index=False, header=False)
    records_path = os.path.join(output_root, "RECORDS")
    with open(records_path, "w", newline="\n") as f:
        p_folders = sampled_df["path"].str.split("/").str[1].unique()
        for p in sorted(p_folders):
            f.write(f"files/{p}/\n")


    # SHA256SUMS
    sha_path = os.path.join(output_root, "SHA256SUMS")
    with open(sha_path, "w", newline="\n") as f:

        def sha256_file(path):
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()

        # 0) Static top-level files
        static_files = ["LICENSE", "LICENSE.txt", "RECORDS"]
        for fname in static_files:
            file_path = os.path.join(output_root, fname)
            if os.path.exists(file_path):
                sha = sha256_file(file_path)
                rel_path_out = os.path.relpath(file_path, output_root).replace("\\", "/")
                f.write(f"{sha}  {rel_path_out}\n")

        # 1) Per-p folders: aggregate SHA + constituent files
        p_folders = sampled_df["path"].str.split("/").str[1].unique()
        for p in sorted(p_folders):
            p_folder_path = os.path.join(output_root, "files", p)

            # Collect all .dat/.hea files under this folder
            file_paths = []
            file_hashes = []
            for root, _, files in os.walk(p_folder_path):
                for fname in sorted(files):
                    if fname.endswith(".dat") or fname.endswith(".hea"):
                        fpath = os.path.join(root, fname)
                        file_paths.append(fpath)
                        file_hashes.append(sha256_file(fpath))

            if file_hashes:
                # Aggregate folder SHA (based on constituent file hashes)
                combined = "".join(file_hashes).encode("utf-8")
                folder_sha = hashlib.sha256(combined).hexdigest()
                rel_folder = os.path.relpath(p_folder_path, output_root).replace("\\", "/")
                f.write(f"{folder_sha}  {rel_folder}/RECORDS\n")

                # Individual file SHAs
                for fpath, sha in zip(file_paths, file_hashes):
                    rel_path_out = os.path.relpath(fpath, output_root).replace("\\", "/")
                    f.write(f"{sha}  {rel_path_out}\n")
                   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample ECG records from PTB-XL or MIMIC-IV-ECG")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["ptbxl", "mimic"])
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to original dataset root")
    parser.add_argument("--output_root", type=str, required=True, help="Path to save sampled subset")
    parser.add_argument("--label_col", type=str, required=True, help="Column name with class labels")
    parser.add_argument("--proportion", type=float, default=0.1, help="Fraction of dataset to sample")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load metadata
    if args.dataset_type == "ptbxl":
        df = pd.read_csv(os.path.join(args.dataset_root, "ptbxl_database.csv"))
    elif args.dataset_type == "mimic":
        df = pd.read_csv(os.path.join(args.dataset_root, "record_list.csv"))

    # Validate label_col
    if args.label_col not in df.columns:
        print(f"Error: Column '{args.label_col}' not found in dataset. Available columns: {list(df.columns)}")
        exit(1)

    sampled_df = sample_ecg_data(
        df=df,
        label_col=args.label_col,
        proportion=args.proportion,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        dataset_type=args.dataset_type,
        random_state=args.random_state
    )

    print(f"Sampling complete! {len(sampled_df)} records selected.")
    print(f"Subset saved to: {args.output_root}")