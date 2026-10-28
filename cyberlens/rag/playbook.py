"""Fixed first steps per fraud family. Time-critical advice never depends on the LLM or on retrieval.
Every number here was checked against the cited document's text."""

MONEY = {"UPI Related Frauds", "DebitCredit Card FraudSim Swap Fraud", "Internet Banking Related Fraud", "Fraud CallVishing",
         "EWallet Related Fraud", "Business Email CompromiseEmail Takeover", "DematDepository Fraud", "Cryptocurrency Fraud",
         "Online Job Fraud", "Online Matrimonial Fraud", "Cheating by Impersonation"}
BANKING = {"UPI Related Frauds", "DebitCredit Card FraudSim Swap Fraud", "Internet Banking Related Fraud", "Fraud CallVishing",
           "EWallet Related Fraud", "Business Email CompromiseEmail Takeover"}
SEXUAL = {"Cyber Blackmailing & Threatening", "Cyber Bullying  Stalking  Sexting", "Unspecified", "Sexual Harassment",
          "Computer Generated CSAM/CSEM"}
ACCOUNT = {"Profile Hacking Identity Theft", "FakeImpersonating Profile", "Email Hacking", "Impersonating Email",
           "Unauthorised AccessData Breach"}
ORG_ATTACK = {"Ransomware Attack", "Ransomware", "Malware Attack", "Data Breach/Theft", "Hacking/Defacement",
              "Denial of Service (DoS)/Distributed Denial of Service (DDOS) attacks", "SQL Injection",
              "Tampering with computer source documents", "Website DefacementHacking", "Damage to computer computer systems etc"}


def step(text, *sources):
    return {"text": text, "sources": list(sources)}


def immediate_steps(subtype, regime):
    sub = (subtype or "").split(" :: ")[-1].replace(" (subtype needs review)", "")
    parent = (subtype or "").split(" :: ")[0]
    steps = []
    if sub in MONEY or parent == "Online Financial Fraud":
        steps.append(step("Call 1930 (national cyber-fraud helpline) or report at cybercrime.gov.in right away. "
                          "Fast reporting lets banks freeze the money before it is moved on.",
                          "pib_ncrp_1930", "pib_blackmail_digital_arrest_alert", "pib_digital_arrest"))
    if sub in BANKING or parent == "Online Financial Fraud":
        if regime == "2017":
            steps.append(step("Report the unauthorised transaction to your bank through its official app, website, "
                              "toll-free number or branch, and keep the complaint number. Reported within 3 working days, "
                              "a third-party fraud means zero liability; within 4-7 working days your loss is capped "
                              "(Rs 5,000 for basic savings (BSBD) accounts, Rs 10,000 for other savings accounts and most "
                              "others, Rs 25,000 for other current accounts and high-limit cards).",
                              "rbi_liability_2017", "rbi_liability_coop_2017"))
            steps.append(step("The bank must credit the disputed amount to your account within 10 working days of your "
                              "report and settle the complaint within 90 days. If you shared your OTP or PIN, you carry "
                              "the loss only until you report it.", "rbi_liability_2017"))
        else:
            steps.append(step("Report to your bank AND to 1930/cybercrime.gov.in within 5 calendar days. A third-party "
                              "fraud reported in time means zero liability, and the bank must decide within 45 days.",
                              "rbi_fraudulent_ebt_2026", "rbi_fraudulent_ebt_2026_pr"))
            steps.append(step("Even if you were tricked into sharing details, you may get a one-time compensation of 85% of "
                              "the loss (up to Rs 25,000, for losses up to Rs 50,000) if you reported to both within 5 days.",
                              "rbi_fraudulent_ebt_2026"))
        steps.append(step("If your bank does not reply within 30 days or you are unhappy with its reply, complain free of "
                          "charge to the RBI Ombudsman at cms.rbi.org.in (helpline 14448) within 90 days.",
                          "rbios_2026", "rbios_2026_faq"))
    if sub == "DebitCredit Card FraudSim Swap Fraud":
        steps.append(step("Ask the card issuer to block the card now; they must block it immediately. No charges may be "
                          "levied on a transaction you dispute as fraud until the dispute is resolved.", "rbi_md_cards", "rbi_cards_faq"))
    if sub in ("Fraud CallVishing", "Cheating by Impersonation") or sub in ("police_digital_arrest", "other_scam"):
        steps.append(step("Report the fraud call, SMS or WhatsApp number on Chakshu (sancharsaathi.gov.in); DoT blocks "
                          "linked numbers and handsets.", "pib_chakshu_reporting", "dot_electricity_kyc", "dot_spoofed_calls"))
        steps.append(step("Police, CBI, NCB or RBI officials do not 'arrest' or question people over video calls or ask for "
                          "money to settle a case. Disconnect and report to 1930.", "pib_blackmail_digital_arrest_alert"))
    if sub in ("DematDepository Fraud", "Cryptocurrency Fraud"):
        steps.append(step("Stop further payments to the app or group. Genuine investments never guarantee returns; check that "
                          "the broker or adviser is SEBI-registered.", "sebi_fake_trading_app", "sebi_beware_investment"))
    if sub in SEXUAL or parent in ("Sexually Obscene material", "Sexually Explicit Act", "Crime Against Women & Children"):
        steps.append(step("Do not pay or send more material. Save screenshots, profile links and chat logs as evidence.",
                          "pib_blackmail_digital_arrest_alert"))
        steps.append(step("Report at cybercrime.gov.in (there is a section for crimes against women and children) and ask "
                          "the platform's grievance officer to take the image down; for intimate or morphed images "
                          "platforms must act within 24 hours.", "pib_ncrp_1930", "it_rules_2021"))
    if sub in ACCOUNT:
        steps.append(step("Recover the account, change passwords, turn on two-factor login, and report the fake or hacked "
                          "profile to the platform and at cybercrime.gov.in.", "pib_ncrp_1930", "it_rules_2021"))
    if sub in ORG_ATTACK or parent == "Cyber Attack/ Dependent Crimes":
        steps.append(step("If this happened to an organisation's systems, it must report the incident to CERT-In within 6 hours "
                          "of noticing it. Individuals should report at cybercrime.gov.in.", "certin_directions_2022", "certin_directions_faq"))
    if not steps:
        steps.append(step("Report the incident at cybercrime.gov.in or call 1930, and keep all evidence.", "pib_ncrp_1930"))
    return steps
