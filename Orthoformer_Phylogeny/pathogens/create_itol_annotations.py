#!/usr/bin/env python3
"""
Create iTOL annotation files for pathogen phylogenetic tree
Based on genus information from metadata
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import colorsys

# Configuration
BASE_DIR = Path(".")
METADATA_FILE = BASE_DIR / "isolates_multi_829_metadata.csv"
TREE_FILE = BASE_DIR / "phylo_tree" / "pathogen_phylogeny_fasttree.nwk"
OUTPUT_DIR = BASE_DIR / "itol_annotations"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("Creating iTOL Annotation Files for Pathogen Phylogenetic Tree")
print("="*80)

# Read metadata
print("\nReading metadata...")
df = pd.read_csv(METADATA_FILE)
print(f"Total entries: {len(df)}")

# Extract genus from Scientific name
print("\nExtracting genus information...")
df['Genus'] = df['Scientific name'].str.split().str[0]

# Count genera
genus_counts = df['Genus'].value_counts()
print(f"\nFound {len(genus_counts)} unique genera:")
for genus, count in genus_counts.head(20).items():
    print(f"  {genus}: {count} isolates")
if len(genus_counts) > 20:
    print(f"  ... and {len(genus_counts) - 20} more genera")

# Generate distinct colors for each genus
def generate_colors(n):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
        value = 0.8 + (i % 2) * 0.15      # Vary brightness slightly
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

# Assign colors to genera (sorted by count for consistency)
genera_sorted = genus_counts.index.tolist()
colors = generate_colors(len(genera_sorted))
genus_to_color = dict(zip(genera_sorted, colors))

# Create mapping from Assembly ID to genus and color
assembly_to_genus = {}
assembly_to_color = {}
for _, row in df.iterrows():
    assembly = row['Assembly']
    genus = row['Genus']
    assembly_to_genus[assembly] = genus
    assembly_to_color[assembly] = genus_to_color[genus]

print(f"\nCreated color mapping for {len(genus_to_color)} genera")

# ============================================================================
# 1. Create Color Strip file (for branch/leaf coloring)
# ============================================================================
print("\n" + "="*80)
print("Creating Color Strip file...")
print("="*80)

color_strip_file = OUTPUT_DIR / "genus_color_strip.txt"
with open(color_strip_file, 'w') as f:
    # Header
    f.write("DATASET_COLORSTRIP\n")
    f.write("SEPARATOR TAB\n")
    f.write("DATASET_LABEL\tGenus\n")
    f.write("COLOR\t#ff0000\n")
    f.write("LEGEND_TITLE\tGenus\n")
    f.write("LEGEND_SHAPES\t" + "\t".join(["1"] * len(genus_to_color)) + "\n")
    f.write("LEGEND_COLORS\t" + "\t".join(genus_to_color.values()) + "\n")
    f.write("LEGEND_LABELS\t" + "\t".join(genus_to_color.keys()) + "\n")
    f.write("STRIP_WIDTH\t50\n")
    f.write("MARGIN\t10\n")
    f.write("SHOW_INTERNAL\t0\n")
    f.write("\n")
    f.write("DATA\n")
    
    # Data lines
    for assembly, color in assembly_to_color.items():
        genus = assembly_to_genus[assembly]
        f.write(f"{assembly}\t{color}\t{genus}\n")

print(f"✓ Color strip file created: {color_strip_file}")
print(f"  Contains {len(assembly_to_color)} entries")

# ============================================================================
# 2. Create Simple Bar Chart file (showing genus distribution)
# ============================================================================
print("\n" + "="*80)
print("Creating Simple Bar Chart file...")
print("="*80)

bar_chart_file = OUTPUT_DIR / "genus_bar_chart.txt"
with open(bar_chart_file, 'w') as f:
    # Header
    f.write("DATASET_SIMPLEBAR\n")
    f.write("SEPARATOR TAB\n")
    f.write("DATASET_LABEL\tGenus Count\n")
    f.write("COLOR\t#0000ff\n")
    f.write("WIDTH\t200\n")
    f.write("MARGIN\t10\n")
    f.write("\n")
    f.write("DATA\n")
    
    # Data lines - show count of each genus
    for assembly, genus in assembly_to_genus.items():
        count = genus_counts[genus]
        color = genus_to_color[genus]
        f.write(f"{assembly}\t{count}\t{color}\n")

print(f"✓ Bar chart file created: {bar_chart_file}")

# ============================================================================
# 3. Create Text Labels file (showing genus names)
# ============================================================================
print("\n" + "="*80)
print("Creating Text Labels file...")
print("="*80)

text_labels_file = OUTPUT_DIR / "genus_text_labels.txt"
with open(text_labels_file, 'w') as f:
    # Header
    f.write("DATASET_TEXT\n")
    f.write("SEPARATOR TAB\n")
    f.write("DATASET_LABEL\tGenus Labels\n")
    f.write("COLOR\t#000000\n")
    f.write("MARGIN\t10\n")
    f.write("SHOW_INTERNAL\t0\n")
    f.write("\n")
    f.write("DATA\n")
    
    # Data lines
    for assembly, genus in assembly_to_genus.items():
        color = genus_to_color[genus]
        f.write(f"{assembly}\t{genus}\t-1\t{color}\tnormal\t1\t0\n")

print(f"✓ Text labels file created: {text_labels_file}")

# ============================================================================
# 4. Create Color Range file (for continuous data like isolation source)
# ============================================================================
print("\n" + "="*80)
print("Creating Isolation Source annotation...")
print("="*80)

# Get unique isolation sources
isolation_sources = df['Isolation_source'].value_counts()
print(f"Found {len(isolation_sources)} unique isolation sources:")
for source, count in isolation_sources.head(10).items():
    print(f"  {source}: {count} isolates")

# Assign colors to isolation sources
source_colors = generate_colors(len(isolation_sources))
source_to_color = dict(zip(isolation_sources.index, source_colors))

isolation_strip_file = OUTPUT_DIR / "isolation_source_color_strip.txt"
with open(isolation_strip_file, 'w') as f:
    # Header
    f.write("DATASET_COLORSTRIP\n")
    f.write("SEPARATOR TAB\n")
    f.write("DATASET_LABEL\tIsolation Source\n")
    f.write("COLOR\t#00ff00\n")
    f.write("LEGEND_TITLE\tIsolation Source\n")
    f.write("LEGEND_SHAPES\t" + "\t".join(["2"] * len(source_to_color)) + "\n")
    f.write("LEGEND_COLORS\t" + "\t".join(source_to_color.values()) + "\n")
    f.write("LEGEND_LABELS\t" + "\t".join(source_to_color.keys()) + "\n")
    f.write("STRIP_WIDTH\t50\n")
    f.write("MARGIN\t10\n")
    f.write("SHOW_INTERNAL\t0\n")
    f.write("\n")
    f.write("DATA\n")
    
    # Data lines
    for _, row in df.iterrows():
        assembly = row['Assembly']
        source = row['Isolation_source']
        color = source_to_color.get(source, '#cccccc')
        f.write(f"{assembly}\t{color}\t{source}\n")

print(f"✓ Isolation source strip file created: {isolation_strip_file}")

# ============================================================================
# 5. Create summary file with genus statistics
# ============================================================================
print("\n" + "="*80)
print("Creating summary statistics file...")
print("="*80)

summary_file = OUTPUT_DIR / "genus_summary.txt"
with open(summary_file, 'w') as f:
    f.write("Genus Summary Statistics\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total isolates: {len(df)}\n")
    f.write(f"Total genera: {len(genus_counts)}\n\n")
    f.write("Genus Distribution:\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Genus':<30} {'Count':>10} {'Percentage':>12} {'Color':>10}\n")
    f.write("-"*80 + "\n")
    
    for genus, count in genus_counts.items():
        percentage = count / len(df) * 100
        color = genus_to_color[genus]
        f.write(f"{genus:<30} {count:>10} {percentage:>11.2f}% {color:>10}\n")

print(f"✓ Summary file created: {summary_file}")

# ============================================================================
# 6. Create README for iTOL upload
# ============================================================================
print("\n" + "="*80)
print("Creating iTOL upload instructions...")
print("="*80)

readme_file = OUTPUT_DIR / "README_ITOL.md"
with open(readme_file, 'w') as f:
    f.write("# iTOL Annotation Files for Pathogen Phylogenetic Tree\n\n")
    f.write("## Overview\n\n")
    f.write(f"- **Tree file**: `{TREE_FILE.name}`\n")
    f.write(f"- **Total isolates**: {len(df)}\n")
    f.write(f"- **Total genera**: {len(genus_counts)}\n")
    f.write(f"- **Tree method**: FastTree (WAG model)\n\n")
    
    f.write("## Generated Annotation Files\n\n")
    f.write("1. **genus_color_strip.txt** - Color strip showing genus for each isolate\n")
    f.write("2. **genus_bar_chart.txt** - Bar chart showing genus abundance\n")
    f.write("3. **genus_text_labels.txt** - Text labels showing genus names\n")
    f.write("4. **isolation_source_color_strip.txt** - Color strip for isolation sources\n")
    f.write("5. **genus_summary.txt** - Summary statistics\n\n")
    
    f.write("## How to Use with iTOL\n\n")
    f.write("### Step 1: Upload Tree\n")
    f.write("1. Go to https://itol.embl.de/\n")
    f.write("2. Click 'Upload' button\n")
    f.write(f"3. Upload the tree file: `{TREE_FILE}`\n")
    f.write("4. Wait for the tree to be processed and displayed\n\n")
    
    f.write("### Step 2: Add Annotations\n")
    f.write("1. Click on the 'Datasets' tab in iTOL\n")
    f.write("2. Drag and drop annotation files one by one:\n")
    f.write("   - Start with `genus_color_strip.txt` (main genus coloring)\n")
    f.write("   - Add `isolation_source_color_strip.txt` (isolation source)\n")
    f.write("   - Optionally add `genus_text_labels.txt` (if tree is not too crowded)\n")
    f.write("   - Optionally add `genus_bar_chart.txt` (for abundance visualization)\n\n")
    
    f.write("### Step 3: Customize Display\n")
    f.write("1. Adjust tree layout (circular, rectangular, etc.)\n")
    f.write("2. Modify branch lengths display\n")
    f.write("3. Adjust font sizes and colors as needed\n")
    f.write("4. Show/hide legends\n\n")
    
    f.write("### Step 4: Export\n")
    f.write("1. Click 'Export' tab\n")
    f.write("2. Choose format (PDF, PNG, SVG)\n")
    f.write("3. Download your annotated tree\n\n")
    
    f.write("## Genus Color Mapping\n\n")
    f.write(f"Total of {len(genus_to_color)} genera with distinct colors:\n\n")
    
    for genus, color in list(genus_to_color.items())[:20]:
        count = genus_counts[genus]
        f.write(f"- **{genus}** ({count} isolates): {color}\n")
    
    if len(genus_to_color) > 20:
        f.write(f"\n... and {len(genus_to_color) - 20} more genera (see genus_summary.txt for complete list)\n")
    
    f.write("\n## Tips\n\n")
    f.write("- For large trees (829 isolates), use circular layout for better visualization\n")
    f.write("- If text labels are too crowded, hide them and rely on color strips\n")
    f.write("- Use the search function in iTOL to find specific isolates\n")
    f.write("- You can combine multiple color strips to show different metadata\n")
    f.write("- Export in SVG format for further editing in vector graphics software\n\n")
    
    f.write("## File Locations\n\n")
    f.write(f"- Tree file: `{TREE_FILE}`\n")
    f.write(f"- Annotation files: `{OUTPUT_DIR}/`\n")
    f.write(f"- Metadata: `{METADATA_FILE}`\n")

print(f"✓ README created: {readme_file}")

# ============================================================================
# Final summary
# ============================================================================
print("\n" + "="*80)
print("✓ All iTOL annotation files created successfully!")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nGenerated files:")
print(f"  1. genus_color_strip.txt")
print(f"  2. genus_bar_chart.txt")
print(f"  3. genus_text_labels.txt")
print(f"  4. isolation_source_color_strip.txt")
print(f"  5. genus_summary.txt")
print(f"  6. README_ITOL.md")
print(f"\nTree file location: {TREE_FILE}")
print(f"\nTo visualize:")
print(f"  1. Go to https://itol.embl.de/")
print(f"  2. Upload: {TREE_FILE}")
print(f"  3. Add datasets from: {OUTPUT_DIR}")
print("\nDone!")

