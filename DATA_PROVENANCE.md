# Data provenance

This repository does not redistribute the raw BAF or ACS files. Users should obtain the data from the public sources below and comply with the applicable terms.

## Bank Account Fraud (BAF)

Public source: [Kaggle — Bank Account Fraud Dataset NeurIPS 2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022).

The frozen public snapshot used by the threshold-transfer audit contains six one-million-row CSV files:

| File | Bytes | SHA-256 |
|---|---:|---|
| `Base.csv` | 213,427,735 | `7BF10A37CE07E72E14C1B09E5EFEE3D27261BAFF4FACC7DA767B0474DCF9B809` |
| `Variant I.csv` | 213,400,445 | `48C637A255D1FA4515DA4286CCB99251412A6D905937EC7C108838DDFC1674BD` |
| `Variant II.csv` | 213,537,521 | `60BCD971CF28779ABB183C3286990CFF9A87275BD8B83D9F6054A396C95CC736` |
| `Variant III.csv` | 252,204,320 | `64693E549CFF7EF802CF043FC7C937839E5929966E33A0852953B3F4C4321339` |
| `Variant IV.csv` | 213,538,370 | `C6CDDEBBAD34FA262A278DEEB7985E7B669A64308D26E480D8EB85BFD08DAC75` |
| `Variant V.csv` | 252,214,315 | `470899BBEF97C32A100528D63E863E1B1036DF9501DF6EF8140E7EC13B7365B4` |

The BAF CSVs are referenced by hash and are not included in this repository.

## American Community Survey (ACS)

Primary source: [U.S. Census Bureau ACS microdata](https://www.census.gov/programs-surveys/acs/microdata.html). The task definition follows [Folktables](https://github.com/zykls/folktables) and the WhyShift ACS setting. Because the official Census download endpoint failed TLS negotiation on the reproduction machine, byte-identical raw state files were acquired from the public [Hugging Face ACSIncome-2018-1-Year mirror](https://huggingface.co/datasets/davidboetius/ACSIncome-2018-1-Year).

The frozen 2018 1-Year PUMS state files are:

| File | State | Bytes | SHA-256 |
|---|---|---:|---|
| `psam_p06.csv` | California | 267,297,811 | `DC2187FC90DF2C5F6B546EE89A2B41C9A97379C9E7136461B6A6C8DE871B43E0` |
| `psam_p72.csv` | Puerto Rico | 19,770,298 | `74436D4FFEE9E5974982AFF45622E49A192DE2F22DE528E2156C7EC13832E244` |
| `psam_p28.csv` | Mississippi | 20,229,762 | `2944DF2DBDC582720746F4A29E8FC47B409B06E65A53C952A5CB1E4691CB0A70` |
| `psam_p15.csv` | Hawaii | 10,096,437 | `57F821DD8C4190FAD8D677948028D75E200A4537C0A2944AFFD9CB50C88DCB8A` |
| `psam_p25.csv` | Massachusetts (unused mirror file) | 49,501,978 | `47BE6D5A3636D3A9A02450239E93D8D25A6D86D25B108F6E6A5A786169F4DD21` |

The ACS files are referenced by hash and are not included in this repository. The analysis uses the official `folktables.ACSIncome` feature definition and filter, unweighted model fitting and endpoint computation for the primary replication, and the state pairs and splits documented in `protocols/external_transfer_protocol_v0.1.md`.
