# Hana Data Labs – Pilot POS Analytics Project

This repository contains an anonymized version of a pilot analytics project for a small business client, built under Hana Data Labs.

The goal of the project is to:
- Clean and analyze POS (point-of-sale) transaction data
- Generate actionable insights about revenue, item performance, and customer behavior
- Build visualizations and dashboards suitable for non-technical stakeholders

All data in this repository is either:
- Synthetic; or
- Aggregated/anonymized so that no individual customer or business is identifiable.

## Repository Structure

- \`data/\` – raw, interim, and processed data (raw client data is **not** committed)
- \`notebooks/\` – exploratory analyses, modeling, and insight generation
- \`src/\` – Python modules for loading, cleaning, and analysis
- \`reports/\` – figures and summary outputs
- \`docs/\` – project documentation and case study writeup

## Environment

To create the conda environment:

```bash
conda env create -f environment.yml
conda activate hana-pilot-ml
