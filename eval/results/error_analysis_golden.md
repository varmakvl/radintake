# Error analysis: 10 worst pages (golden, ocr_first)

Pages scored: 145. Worst pages by tier: phone_photo 6, poor_scan 4.

| # | Doc | Page | Tier | Format | Handwriting | Mean CER | Failures |
|---|---|---|---|---|---|---|---|
| 1 | D0178 | 1 | phone_photo | tiff | yes | 0.8 | patient_id: omission; indication: omission; comparison: omission; findings: omission; patient_name: PHI miss; referring_physician: PHI miss; patient_id: PHI miss; date_of_birth: PHI miss; accession_number: PHI miss |
| 2 | D0097 | 1 | phone_photo | jpg | no | 0.8333 | accession_number: omission; indication: omission; comparison: omission; findings: omission; impression: omission; patient_name: PHI miss; referring_physician: PHI miss; accession_number: PHI miss; radiologist: PHI miss |
| 3 | D0290 | 2 | poor_scan | tiff | yes | 0.4444 | accession_number: layout: bleed; impression: omission; patient_name: PHI miss; patient_id: PHI miss; accession_number: PHI miss; technologist: PHI miss; radiologist: PHI miss |
| 4 | D0322 | 1 | phone_photo | jpg | no | 0.8636 | patient_id: omission; accession_number: omission; indication: OCR misread; impression: hallucination; patient_name: PHI miss; technologist: PHI miss; radiologist: PHI miss |
| 5 | D0051 | 2 | phone_photo | pdf | yes | 0.3333 | impression: hallucination; patient_name: PHI miss; patient_id: PHI miss; accession_number: PHI miss; technologist: PHI miss; radiologist: PHI miss |
| 6 | D0057 | 2 | phone_photo | tiff | no | 0.3333 | impression: omission; patient_name: PHI miss; patient_id: PHI miss; accession_number: PHI miss; technologist: PHI miss; radiologist: PHI miss |
| 7 | D0108 | 1 | phone_photo | jpg | no | 0.8 | accession_number: omission; comparison: omission; findings: omission; impression: omission; accession_number: PHI miss; technologist: PHI miss; radiologist: PHI miss |
| 8 | D0060 | 1 | poor_scan | png | no | 0.7733 | accession_number: omission; indication: hallucination; comparison: omission; findings: hallucination; impression: omission; patient_name: PHI miss; referring_physician: PHI miss; radiologist: PHI miss |
| 9 | D0058 | 1 | poor_scan | pdf | no | 0.453 | indication: OCR misread; comparison: omission; findings: hallucination; impression: hallucination; patient_name: PHI miss; referring_physician: PHI miss; technologist: PHI miss; radiologist: PHI miss |
| 10 | D0245 | 1 | poor_scan | tiff | yes | 0.6 | indication: omission; comparison: omission; findings: omission; patient_name: PHI miss; referring_physician: PHI miss; accession_number: PHI miss |

## Failure categories

| Category | Worst 10 pages | All scored pages |
|---|---|---|
| PHI miss | 40 | 149 |
| OCR misread | 2 | 74 |
| hallucination | 6 | 51 |
| omission | 24 | 50 |
| layout: bleed | 1 | 22 |
| layout: truncated | 0 | 9 |
| handwriting | 0 | 8 |
| wrong field | 0 | 1 |
