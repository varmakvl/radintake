# Error analysis: 10 worst pages (dev, ocr_first)

Pages scored: 267. Worst pages by tier: phone_photo 8, poor_scan 2.

| # | Doc | Page | Tier | Format | Handwriting | Mean CER | Failures |
|---|---|---|---|---|---|---|---|
| 1 | D0294 | 1 | phone_photo | jpg | yes | 1.0 | patient_id: omission; accession_number: omission; indication: omission; comparison: omission; findings: omission; impression: omission; patient_name: PHI miss; referring_physician: PHI miss; patient_id: PHI miss; date_of_birth: PHI miss; accession_number: PHI miss; technologist: PHI miss; radiologist: PHI miss |
| 2 | D0313 | 1 | phone_photo | jpg | yes | 1.0 | patient_id: omission; accession_number: omission; indication: omission; comparison: omission; findings: omission; impression: omission; patient_name: PHI miss; referring_physician: PHI miss; patient_id: PHI miss; date_of_birth: PHI miss; accession_number: PHI miss; technologist: PHI miss; radiologist: PHI miss |
| 3 | D0090 | 1 | phone_photo | jpg | no | 1.0 | patient_id: omission; accession_number: omission; indication: omission; comparison: omission; findings: omission; impression: omission; patient_name: PHI miss; patient_id: PHI miss; date_of_birth: PHI miss; accession_number: PHI miss; technologist: PHI miss; radiologist: PHI miss |
| 4 | D0100 | 1 | poor_scan | png | yes | 0.7969 | patient_id: omission; accession_number: omission; indication: handwriting; comparison: hallucination; findings: hallucination; impression: omission; patient_name: PHI miss; patient_id: PHI miss; date_of_birth: PHI miss; accession_number: PHI miss; technologist: PHI miss |
| 5 | D0079 | 1 | phone_photo | jpg | yes | 0.7168 | patient_id: omission; accession_number: omission; comparison: hallucination; findings: hallucination; impression: omission; patient_name: PHI miss; referring_physician: PHI miss; date_of_birth: PHI miss; technologist: PHI miss; radiologist: PHI miss |
| 6 | D0208 | 2 | poor_scan | pdf | no | 0.6667 | patient_id: omission; impression: hallucination; patient_name: PHI miss; patient_id: PHI miss; accession_number: PHI miss; technologist: PHI miss; radiologist: PHI miss |
| 7 | D0019 | 1 | phone_photo | pdf | yes | 0.5 | indication: omission; comparison: omission; patient_name: PHI miss; referring_physician: PHI miss; patient_id: PHI miss; date_of_birth: PHI miss; accession_number: PHI miss |
| 8 | D0089 | 1 | phone_photo | tiff | no | 1.0 | patient_id: omission; accession_number: omission; indication: omission; comparison: omission; findings: omission; patient_name: PHI miss; patient_id: PHI miss; date_of_birth: PHI miss |
| 9 | D0218 | 1 | phone_photo | jpg | no | 1.0 | patient_id: omission; accession_number: omission; indication: omission; comparison: omission; findings: omission; impression: omission; patient_name: PHI miss; technologist: PHI miss; radiologist: PHI miss |
| 10 | D0275 | 1 | phone_photo | jpg | no | 0.7353 | patient_id: omission; indication: omission; comparison: hallucination; findings: omission; impression: hallucination; patient_name: PHI miss; referring_physician: PHI miss; technologist: PHI miss; radiologist: PHI miss |

## Failure categories

| Category | Worst 10 pages | All scored pages |
|---|---|---|
| PHI miss | 50 | 290 |
| OCR misread | 0 | 143 |
| omission | 41 | 106 |
| hallucination | 7 | 99 |
| layout: bleed | 0 | 42 |
| handwriting | 1 | 19 |
| layout: truncated | 0 | 19 |
| wrong field | 0 | 4 |
