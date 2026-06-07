# Sample Seed Datasets

Small example datasets for getting started with llm-synth.
Each file is 500 rows — enough to run the full pipeline and explore outputs.
To generate at scale, replace with your full seed CSV.

---

## Included

| File | Condition | Rows | Target | Source |
|---|---|---|---|---|
| `diabetes_sample.csv` | Diabetes hospital readmissions | 500 | `readmitted` (NO / >30 / <30) | UCI ML Repository — Diabetes 130-US hospitals |

---

## How to use a sample

```bash
# Run the full pipeline on the diabetes sample
llm-synth run \
    --seed data/samples/diabetes_sample.csv \
    --condition diabetes \
    --rows 500
```

The domain config (`domains/diabetes.yaml`) is pre-configured for this dataset.

---

## Adding more datasets

### CHF — Heart Failure (299 rows)

Download from Kaggle:
https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data

Save as `data/samples/chf_sample.csv`, then create `domains/chf.yaml`
following the guide in the main README.

Key columns: `age`, `ejection_fraction`, `serum_creatinine`,
`serum_sodium`, `platelets`, `time`, `DEATH_EVENT`

### CKD — Chronic Kidney Disease

Download from UCI ML Repository:
https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

### Bring your own

Any CSV with a clear target column works. Follow the
**Running for a New Disease Area** section in the main README.
