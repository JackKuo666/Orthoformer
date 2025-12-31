#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert antiSMASH GenBank (.gbk) files to GFF3 format with full features.
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

from Bio import SeqIO

# Qualifiers to ignore (typically long or not useful in GFF)
IGNORE_QUALIFIERS = {
    # "translation",
    # "orig_protein_id",
    # "inference",
}


def gff3_escape(val: str) -> str:
    """Escape special characters for GFF3 attribute values."""
    if val is None:
        return ""
    return str(val)


def strand_to_char(loc) -> str:
    """Convert Biopython strand to GFF3 strand character."""
    if loc is None or loc.strand is None:
        return "."
    return "+" if loc.strand >= 0 else "-"


def feature_phase(feature) -> str:
    """
    Calculate the phase for CDS features.
    Phase = (codon_start - 1), where codon_start is typically 1, 2, or 3.
    """
    if feature.type != "CDS":
        return "."
    codon_start = feature.qualifiers.get("codon_start", ["1"])
    try:
        cs = int(codon_start[0])
        return str((cs - 1) % 3)
    except Exception:
        return "0"


def qualifiers_to_attributes(q: dict) -> str:
    """
    Convert qualifiers dictionary to GFF3 attributes string.
    Multiple values are joined with commas.
    """
    items = []
    for k, v in q.items():
        if k in IGNORE_QUALIFIERS:
            continue
        if isinstance(v, (list, tuple)):
            val = ",".join(gff3_escape(x) for x in v if x is not None)
        else:
            val = gff3_escape(v)
        if val == "":
            continue
        items.append(f"{k}={val}")
    return ";".join(items)


def build_gene_index(records):
    """
    Build an index mapping locus_tag/gene_name to gene_id for each sequence.
    Used to establish Parent relationships for CDS features.
    """
    gene_idx = defaultdict(dict)
    for rec in records:
        seqid = rec.id
        seen = 0
        for f in rec.features:
            if f.type != "gene":
                continue
            locus = f.qualifiers.get("locus_tag", [])
            gname = f.qualifiers.get("gene", [])
            if locus:
                gid = f"gene:{locus[0]}"
                key = locus[0]
            elif gname:
                gid = f"gene:{gname[0]}"
                key = gname[0]
            else:
                seen += 1
                gid = f"gene:auto_{seen}"
                key = gid
            gene_idx[seqid][key] = gid
    return gene_idx


def guess_parent_for_cds(feature, gene_idx_for_seq):
    """Find the parent gene ID for a CDS feature using locus_tag or gene name."""
    locus = feature.qualifiers.get("locus_tag", [])
    gname = feature.qualifiers.get("gene", [])
    if locus and locus[0] in gene_idx_for_seq:
        return gene_idx_for_seq[locus[0]]
    if gname and gname[0] in gene_idx_for_seq:
        return gene_idx_for_seq[gname[0]]
    return None


def write_gff3(in_path: Path, out_fh):
    """
    Convert a GenBank file to GFF3 format and write to the output file handle.

    Args:
        in_path: Path to the input GenBank file.
        out_fh: Output file handle for writing GFF3.
    """
    records = list(SeqIO.parse(str(in_path), "genbank"))
    if not records:
        raise SystemExit(f"[ERROR] No records found in {in_path}")

    gene_idx = build_gene_index(records)

    print("##gff-version 3", file=out_fh)
    for rec in records:
        print(f"##sequence-region {rec.id} 1 {len(rec.seq)}", file=out_fh)

    for rec in records:
        seqid = rec.id
        source_default = "GenBank"

        # Sort features by start position (Biopython uses 0-based; GFF3 uses 1-based)
        features_sorted = sorted(
            rec.features,
            key=lambda f: (int(f.location.start), int(f.location.end))
        )

        auto_gene_counter = 0

        for f in features_sorted:
            ftype = f.type
            start = int(f.location.start) + 1
            end = int(f.location.end)
            strand = strand_to_char(f.location)
            score = "."
            phase = feature_phase(f)

            # Use 'antismash' as source for antiSMASH-specific features
            if ftype.startswith("aS_"):
                source = "antismash"
            else:
                source = source_default

            attrs = {}

            if ftype == "gene":
                gid = None
                if "locus_tag" in f.qualifiers:
                    gid = f"gene:{f.qualifiers['locus_tag'][0]}"
                    attrs["ID"] = gid
                    attrs["Name"] = f.qualifiers["locus_tag"][0]
                elif "gene" in f.qualifiers:
                    gid = f"gene:{f.qualifiers['gene'][0]}"
                    attrs["ID"] = gid
                    attrs["Name"] = f.qualifiers["gene"][0]
                else:
                    auto_gene_counter += 1
                    gid = f"gene:auto_{auto_gene_counter}"
                    attrs["ID"] = gid
                other = qualifiers_to_attributes(f.qualifiers)
                if other:
                    attrs["Note"] = gff3_escape(other)

            elif ftype == "CDS":
                parent = guess_parent_for_cds(f, gene_idx.get(seqid, {}))
                if parent:
                    attrs["Parent"] = parent
                if "protein_id" in f.qualifiers:
                    attrs["ID"] = f"CDS:{f.qualifiers['protein_id'][0]}"
                    attrs["Name"] = f.qualifiers["protein_id"][0]
                elif "locus_tag" in f.qualifiers:
                    attrs["ID"] = f"CDS:{f.qualifiers['locus_tag'][0]}"
                    attrs["Name"] = f.qualifiers["locus_tag"][0]
                other = qualifiers_to_attributes(f.qualifiers)
                if other:
                    attrs["Note"] = gff3_escape(other)

            else:
                # Other features (including antiSMASH aS_* features)
                if "locus_tag" in f.qualifiers:
                    attrs["Name"] = f.qualifiers["locus_tag"][0]
                elif "gene" in f.qualifiers:
                    attrs["Name"] = f.qualifiers["gene"][0]

                if ftype.startswith("aS_"):
                    attrs["ID"] = f"{ftype}:{start}-{end}"
                if "product" in f.qualifiers:
                    attrs["product"] = f.qualifiers["product"][0]

                other = qualifiers_to_attributes(f.qualifiers)
                if other:
                    attrs["Note"] = gff3_escape(other)

            attr_str = ";".join(f"{k}={gff3_escape(v)}" for k, v in attrs.items() if v)

            print(
                "\t".join([
                    seqid,
                    source,
                    ftype,
                    str(start),
                    str(end),
                    score,
                    strand,
                    phase,
                    attr_str if attr_str else ".",
                ]),
                file=out_fh
            )


def main():
    parser = argparse.ArgumentParser(
        description="Convert antiSMASH GenBank files to GFF3 format."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input .gbk file or directory (matches *_genomic.gbk in directory).")
    parser.add_argument("-o", "--output", default="-",
                        help="Output GFF3 file path or directory. Default: stdout.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.is_file():
        if out_path == Path("-"):
            write_gff3(in_path, sys.stdout)
        else:
            with open(out_path, "w", encoding="utf-8") as fh:
                write_gff3(in_path, fh)

    elif in_path.is_dir():
        gbks = sorted(in_path.glob("*_genomic.gbk"))
        if not gbks:
            sys.exit(f"[ERROR] No *_genomic.gbk found in directory: {in_path}")

        if out_path == Path("-"):
            sys.exit("[ERROR] Batch mode requires an output directory (-o OUT_DIR).")

        out_path.mkdir(parents=True, exist_ok=True)
        for gbk in gbks:
            out_file = out_path / (gbk.stem + ".gff3")
            with open(out_file, "w", encoding="utf-8") as fh:
                write_gff3(gbk, fh)
        print(f"[OK] Wrote {len(gbks)} GFF3 files to {out_path}", file=sys.stderr)
    else:
        sys.exit(f"[ERROR] Input path not found: {in_path}")


if __name__ == "__main__":
    main()
