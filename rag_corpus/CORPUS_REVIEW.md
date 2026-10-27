# RAG corpus review (2026-10-27)

I read the text of every downloaded file (pdftotext / HTML text) to check two things: is it the real document, and does it give a victim an official right, deadline or step that Agent 3 can cite?

## Verdict
The corpus is solid for **money-loss frauds** (UPI, cards, net banking, wallets, loan apps). It had three real gaps, now fixed in the manifest (re-run `download.ps1` to fetch them):
1. **Wrong IT Act version.** The file was the original 2000 text, which has no s.66C (identity theft), 66D (cheating by personation online), 66E, 67A or 67B. Those are exactly the sections a complaint needs. Replaced with India Code's updated Act.
2. **No criminal law beyond the IT Act.** Cheating, extortion and stalking now sit in the Bharatiya Nyaya Sanhita 2023 (it replaced the IPC on 1 Jul 2024). Added BNS, plus BNSS 2023 for the zero FIR / e-FIR right.
3. **Nothing for non-financial victims** (sextortion, morphed photos, fake profiles). Added the IT Intermediary Rules 2021: platforms must take down intimate or morphed images within 24 hours of the victim's complaint.

Also: the 1930/NCRP guide is still blocked, so I added a PIB release on NCRP + 1930 as a stand-in. One SEBI PDF was corrupt (re-download queued).

## What each group actually gives you
| Role | Documents | What a victim gets |
|---|---|---|
| Rights (bank must...) | RBI liability 2017 + co-op, RBI 2026 amendment, TAT 2019, PPI MD s.17, Cards MD + FAQ, Digital Lending 2025, KYC FAQ, IT Rules 2021 | Zero/limited liability, refund deadlines, compensation, data and recovery rules |
| Procedure (where/when to complain) | RB-IOS 2026 + FAQ (2021 for old cases), ODR 2020, PIB NCRP/1930, Chakshu releases, BNSS | Who to approach, in what order, with what deadline |
| Law (for the complaint draft) | IT Act (updated), BNS 2023 | Section numbers |
| Prevention / scam patterns | BE(A)WARE, NPCI Safety Shield, SEBI scam sheets, PIB digital-arrest alerts, DoT releases | Red flags, "what to do now" tips |
| Background (low value for victims) | 2026 feedback annex, KYC MD (107 pages), Digital Payment Security MD, CERT-In directions | Bank or organisation obligations; keep out of default retrieval |
| Excluded | Alert Family (image-only comic, ~570 words of text), Sanchar Saathi help page (mostly navigation) | Not worth indexing as is |

## Facts I checked in the text that the agent must get right
- **Until 31 Dec 2026 (2017 circular):** zero liability if the bank is at fault, or if a third-party breach is reported within **3 working days**. 4 to 7 days: liability capped at Rs 5,000 / 10,000 / 25,000 by account type. Shadow credit within **10 working days**, resolution within **90 days**, burden of proof on the bank. If the customer shared OTP/PIN, they bear the loss until they report it.
- **From 1 Jan 2027 (2026 amendment):** covers fraud where credentials were obtained by deception, or the customer was coerced. Zero liability for third-party breach reported within **5 calendar days**. Resolution within **45 days** (60 cross-border). New **one-time compensation**: if the customer was negligent and lost up to Rs 50,000, they get 85% of the net loss, capped at Rs 25,000, *if* they reported to 1930/NCRP **and** the bank within 5 days. Ignoring a clear scam warning counts as negligence.
- **Grey zone:** a victim who was tricked into sending the money themselves (authorised push payment) is not clearly covered, even in the 2026 rules. The agent should say "may not be covered; report anyway" rather than promise a refund.
- **Ombudsman (RB-IOS 2026):** complain to the bank first. After 30 days with no reply, or after an unsatisfactory reply, go to the Ombudsman within 90 days. Free; cms.rbi.org.in, toll-free 14448. Up to Rs 30 lakh for consequential loss and Rs 3 lakh for harassment. A police case does *not* bar an Ombudsman complaint.
- **ODR 2020 is for failed transactions only** (money debited but not credited), not fraud. Reversal deadline T+1 day for UPI/IMPS (TAT 2019), with Rs 100/day compensation after that.
- **Loan apps:** apps may not read contacts, files or call logs. Recovery agents must be named to the borrower in advance. Grievance officer first, then the Ombudsman after 30 days. The lender is liable for the app's conduct.

## Before chunking
- Use `role` and `priority` in the manifest: index core + supporting; leave optional/background out of default retrieval or down-weight them.
- Store `effective_from` / `effective_to` on chunks and filter by the date of the incident.
- Hindi headers in RBI PDFs extract as garbled characters. Strip non-Latin lines from the bilingual RBI PDFs, or use the English-only pages.
- Remaining gaps you may want later: the cybercrime.gov.in 1930 guide (browser download), the SEBI SCORES complaint process for demat fraud, and the Consumer Protection Act (for e-commerce fraud).
