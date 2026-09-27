# Feature Report: What Lenders Use to Assess Repayment Risk

Purpose: identify the features regulators, lenders, and peer-reviewed research treat as evidence of repayment ability, then state whether `application_{train|test}.csv` can build each one. Each buildable feature becomes an EDA question now and a candidate engineered feature in the data preparation stage.

## Sources

| # | Source | Type |
|---|--------|------|
| S1 | CFPB, 12 CFR 1026.43(c)(2), Ability-to-Repay underwriting factors | Regulation |
| S2 | CFPB, 12 CFR 1002.6 (Regulation B), rules on evaluating applications | Regulation |
| S3 | World Bank / ICCR, *Credit Scoring Approaches Guidelines* (2019) | Multilateral guidance |
| S4 | Federal Reserve, CFPB, FDIC, NCUA, OCC, *Interagency Statement on the Use of Alternative Data in Credit Underwriting* (Dec 2019) | Regulator guidance |
| S5 | EU AI Act, Annex III, point 5(b): creditworthiness scoring of natural persons is high-risk | Regulation |
| S6 | Berg, Burg, Gombović and Puri (2020), "On the Rise of FinTechs: Credit Scoring Using Digital Footprints," *Review of Financial Studies* 33(7) | Peer-reviewed |
| S7 | CFPB Office of Research, *Data Point: Credit Invisibles* (2015) | Regulator research |
| S8 | FinTech Futures, "Case study: Home Credit" (lender disclosure: 6M decisions/month, ~20% repeat borrowers) | Lender disclosure |

## Feature map

| Industry feature | Evidence | Can this data build it? | Columns | EDA question |
|---|---|---|---|---|
| Payment-to-income (PTI) | S1 factors 1, 3, 6; S3 | Yes | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Is income believable? Is PTI predictive? |
| Loan-to-value (LTV) | S3 (collateral value) | Yes | `AMT_CREDIT / AMT_GOODS_PRICE` | How often is goods price missing? Does LTV > 1 occur, and does it predict default? |
| Loan term | S3 (maturity) | Yes, derived | `AMT_CREDIT / AMT_ANNUITY` (approximate months) | Do longer terms carry more risk? |
| Employment status and tenure | S1 factor 2; S3 (job tenure) | Yes, with a data quality caveat | `DAYS_EMPLOYED`, `NAME_INCOME_TYPE`, `OCCUPATION_TYPE` | Does tenure predict default, and what is the 365243 value? |
| Credit history / bureau score | S1 factor 7; S3 | Partially: only as normalized external scores and inquiry counts | `EXT_SOURCE_1/2/3`, `AMT_REQ_CREDIT_BUREAU_*` | How much do external scores separate risk, and how much does missingness limit them? |
| Recent credit inquiries | S3 ("number of credit inquiries") | Yes | `AMT_REQ_CREDIT_BUREAU_HOUR…YEAR` | Does recent credit seeking signal risk? |
| Residential and identity stability | S3 (length at address) | Yes, as proxies | `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `DAYS_LAST_PHONE_CHANGE`, `REG_*`/`LIVE_*` mismatch flags | Do stability indicators relate to default? |
| Thin-file status | S7, S6 | Proxy only | Missing `EXT_SOURCE_*` and missing bureau inquiry counts | What share of applicants are thin-file, and do they default differently? |
| Alternative data (cash flow, telco, digital footprint) | S4, S6 | **No.** Not present in the application tables | n/a | Out of scope; noted as a later-phase opportunity in the BPS |
| Protected characteristics | S2 (sex prohibited; age allowed only if elderly applicants are not penalized); S3; S5 | Present in data, must be governed | `CODE_GENDER`, `DAYS_BIRTH`, `NAME_FAMILY_STATUS` | How strongly do protected attributes relate to default, and which predictors act as proxies? |

## Notes

- AI recall of Kaggle notebooks is not treated as industry research. Every row above traces to a regulator, multilateral body, lender disclosure, or peer-reviewed paper.
- S4 and S5 frame the governance constraint: any model that reaches production must be explainable to a declined applicant and documented as a high-risk system in EU markets.
