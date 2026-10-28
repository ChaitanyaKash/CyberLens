# CyberLens RAG corpus (official sources)

37 official documents for Agent 3 (guidance + complaint draft). 20 are marked `core`; start with those if short on time.

Files:
- `manifest.json` / `manifest.csv`: id, title, issuer, date, URL, local file, priority, the CyberLens subtypes each doc serves, and a short note.
- `subtype_to_docs.json`: the retrieval filter. For a predicted subtype, search those doc ids plus `_applies_to_every_subtype`.
- `download.ps1`: run on the laptop from this folder. It saves to `pdf/` and `html/`, checks each PDF really starts with `%PDF` (RBI sometimes serves a "are you human" page instead), and writes `failed_downloads.csv` for anything you need to save from the browser by hand.

Why the cloud didn't download them: rbi.org.in, cybercrime.gov.in, sancharsaathi.gov.in, npci.org.in, indiacode.nic.in all refuse connections from the cloud proxy, and RBI PDFs also sit behind a bot check.

## Things to know before building
1. **Customer liability rules change on 1 Jan 2027.** The July 2017 circular (zero liability if reported within 3 working days) still applies to transactions up to 31 Dec 2026. RBI's Responsible Business Conduct Third Amendment Directions (24 Jun 2026) replace it from 1 Jan 2027: wider fraud scope, burden of proof on the bank, shorter bank deadlines, and compensation for small-value fraud. Store an `effective_from` / `effective_to` on chunks and have the agent pick by transaction date.
2. **Ombudsman: use the 2026 scheme.** RB-IOS 2026 came into force on 1 Jul 2026 and replaces RB-IOS 2021 (keep 2021 only for older complaints, tagged superseded). Wait 30 days after complaining to the bank, unless the bank replies and you are unhappy.
3. HTML pages (PIB, NPCI, SEBI, Sanchar Saathi): if a download is a JS shell or a bot check, open in a browser and use Print > Save as PDF.
4. Non-financial subtypes (sextortion, CSAM, stalking, hacking) mostly route to the cybercrime.gov.in manual, IT Act, CERT-In, and 1930. That is intentional; the RBI docs are not relevant there.

Chunking suggestion: ~500 tokens with ~50 overlap, keep `id`, `title`, `issuer`, `date`, page number and section heading on every chunk so answers can cite "RBI circular RBI/2017-18/15, para 6".
