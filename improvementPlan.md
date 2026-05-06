# AFAD Text Model Improvement Plan (v3 - Indian Context Fix)

## Problem Statement
The model failed to detect:
"send 1 lakh" → predicted Safe

Reason:
- Dataset lacks Indian financial language
- No exposure to local scam patterns
- TF-IDF cannot infer meaning of "lakh" without training examples

---

## Objective
Improve real-world scam detection by:
- Adding Indian-English + Manglish patterns
- Strengthening financial + urgency detection
- Adding rule-based safety layer

---

## Phase 1 — Dataset Expansion (HIGH PRIORITY)

### 1.1 Add Indian Financial Terms

Add attack samples with:

- lakh / lakhs
- crore
- rs / rupees
- cash transfer

Examples:

- send 1 lakh urgently
- need 50k immediately
- transfer 2 lakh now
- send money asap
- emergency need cash

---

### 1.2 Add Payment Platform Terms

Include:

- gpay
- phonepe
- paytm
- UPI

Examples:

- gpay me 5000 now
- send via UPI urgently
- phonepe me asap

---

### 1.3 Add Manglish Variants (IMPORTANT)

Examples:

- da send 1 lakh urgent
- bro cash venam urgently
- money ayachu thaa
- urgent ayittu paisa venam
- send cash da fast

⚠ Keep labels:
→ All = Attack

---

### 1.4 Dataset Target

Add:

- +800 to +1200 new attack samples

Final goal:

Safe: ~5000  
Attack: ~2500+

---

## Phase 2 — Feature Engineering Upgrade

### 2.1 Add Keyword Flags

Create manual features:

- has_money_term
- has_urgency
- has_payment_term

---

### 2.2 Keyword Lists

Money terms:
- lakh, crore, rs, rupees, cash, money, amount

Urgency terms:
- urgent, asap, immediately, fast, now

Payment terms:
- gpay, phonepe, paytm, upi

---

### 2.3 Feature Example

Input:
"send 1 lakh urgently"

Output:
money=1, urgency=1, payment=0

---

### 2.4 Combine Features

Final input:

TF-IDF features  
+  
Keyword flags

---

## Phase 3 — Model Update

Use:

LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

Reason:
- Handles imbalance better
- Improves attack detection recall

---

## Phase 4 — Rule-Based Safety Layer (CRITICAL FIX)

Add override rule:

IF:
- money_term == TRUE
AND
- urgency == TRUE

THEN:
→ Force prediction = Attack

---

### Example

"send 1 lakh now"

Even if model says Safe → Override → Attack

---

## Phase 5 — Threshold Tuning

Instead of fixed 0.5:

- > 0.7 → Attack
- 0.4 – 0.7 → Suspicious
- < 0.4 → Safe

---

## Phase 6 — Testing Cases

Test with:

| Input | Expected |
|------|---------|
| send 1 lakh | Attack |
| gpay me now | Attack |
| bro money venam urgent | Attack |
| meeting tomorrow | Safe |
| hello how are you | Safe |

---

## Phase 7 — Expected Improvements

- Detect Indian financial scams correctly
- Reduce false negatives
- Improve real-world reliability
- Stronger demo performance

---

## Final Outcome

System becomes:

Localized + Context-aware + Safer

Instead of:

Textbook-trained + Easily fooled
