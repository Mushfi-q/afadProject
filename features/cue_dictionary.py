"""
AFAD - Social Engineering Cue Dictionary
Contains lists of manipulation words used for custom feature extraction.
"""

FAMILIARITY_CUES = [
    "bro",
    "friend",
    "it's me",
    "remember me",
    "long time",
    "my old number",
    "new number",
    "i lost my phone"
]

URGENCY_CUES = [
    "urgent",
    "immediately",
    "asap",
    "right now",
    "quickly",
    "now",
    "today",
    "fast"
]

EMOTIONAL_CUES = [
    "please help",
    "i am in trouble",
    "emergency",
    "don't tell anyone",
    "i need your help"
]

AUTHORITY_CUES = [
    "this is your boss",
    "bank manager",
    "police officer",
    "hr department",
    "manager here"
]

# NEW: Phase 2.2 Keyword Lists
MONEY_TERMS = [
    "lakh", "crore", "rs", "rupees", "cash", "money", "amount"
]

URGENCY_TERMS = [
    "urgent", "asap", "immediately", "fast", "now"
]

PAYMENT_TERMS = [
    "gpay", "phonepe", "paytm", "upi"
]
