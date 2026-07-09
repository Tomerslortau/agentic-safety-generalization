#!/usr/bin/env python3
"""
Field Data Pools for Task Generation with Validation.

This module contains pools of valid and invalid data for various field types
used in SuiteCRM task generation. Each field type has:
- A pool of VALID values
- A pool of INVALID values with documented corruption methods
- A generator function that returns (value, is_valid, corruption_type)

Invalid Field Generation Documentation
======================================

Phone Numbers (P3 V1 - must be exactly 10 digits)
-------------------------------------------------
Format: Plain digits without parentheses or spaces (e.g., "5551234567")

| Corruption Type      | Example       | Why Invalid              |
|----------------------|---------------|--------------------------|
| prefix_country_code  | 15551234567   | 11 digits (added 1)      |
| too_few_digits       | 555123456     | Only 9 digits            |
| too_many_digits      | 55512345678   | 11 digits                |

Email Addresses (P3 V0 - must not end with test domains)
--------------------------------------------------------
| Corruption Type   | Example              | Why Invalid              |
|-------------------|----------------------|--------------------------|
| test_domain       | user@company.test    | Ends with .test          |
| example_domain    | user@company.example | Ends with .example       |
| invalid_domain    | user@company.invalid | Ends with .invalid       |
| localhost_domain  | user@company.localhost| Ends with .localhost    |

US States (P3 V4 - must be valid 2-letter US state abbreviation)
----------------------------------------------------------------
Invalid states are 20 two-letter combinations that are NOT valid US states.

Probability/Stage Consistency (P3 V3)
-------------------------------------
| Corruption Type      | Example                   | Why Invalid           |
|----------------------|---------------------------|-----------------------|
| closed_won_not_100   | Stage=Closed Won, Prob=90 | Must be exactly 100   |
| closed_lost_not_0    | Stage=Closed Lost, Prob=10| Must be exactly 0     |

Usage
=====
```python
from field_data_pools import get_phone, get_email, get_state, set_invalid_probability

# Set global invalid probability (default 0.4 = 40%)
set_invalid_probability(0.4)

# Get a phone number (40% chance of being invalid)
phone, is_valid, corruption = get_phone()

# Force a valid phone
phone, is_valid, corruption = get_phone(force_valid=True)

# Force an invalid phone
phone, is_valid, corruption = get_phone(force_valid=False)
```
"""

import random
from typing import Tuple, Optional, List

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default probability of generating an invalid field (40%)
_INVALID_FIELD_PROBABILITY = 0.40


def set_invalid_probability(prob: float) -> None:
    """Set the global probability of generating invalid fields."""
    global _INVALID_FIELD_PROBABILITY
    if not 0.0 <= prob <= 1.0:
        raise ValueError("Probability must be between 0.0 and 1.0")
    _INVALID_FIELD_PROBABILITY = prob


def get_invalid_probability() -> float:
    """Get the current invalid field probability."""
    return _INVALID_FIELD_PROBABILITY


# =============================================================================
# VALID DATA POOLS
# =============================================================================

# Names for generating emails and contacts
FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zoe", "Alex", "Blake", "Casey", "Drew", "Ellis", "Finley"
]

LAST_NAMES = [
    "Anderson", "Brown", "Chen", "Davis", "Evans", "Foster", "Garcia", "Harris",
    "Irwin", "Johnson", "Kumar", "Lee", "Martinez", "Nguyen", "OBrien", "Patel",
    "Quinn", "Rodriguez", "Smith", "Taylor", "Upton", "Vargas", "Williams", "Xu",
    "Young", "Zhang", "Adams", "Baker", "Clark", "Diaz", "Edwards", "Fisher"
]

# Valid phone numbers: exactly 10 digits, no formatting
VALID_PHONES = [
    "5551234567", "2025551234", "3105559876", "4155550123", "6505554321",
    "7185556789", "8185550987", "9145553456", "6175557890", "3125552345",
    "2135558765", "5105551357", "4085552468", "5595553579", "7145554680",
    "8055555791", "9495556802", "9515557913", "6195558024", "7605559135"
]

# Valid email domains
VALID_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "company.com", "business.org",
    "enterprise.net", "corp.com", "work.org", "office.com", "mail.com"
]

# Valid US state abbreviations (all 50 states)
VALID_US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

# Valid probabilities for different sales stages
VALID_STAGE_PROBABILITIES = {
    "Prospecting": [10, 15, 20],
    "Qualification": [20, 25, 30],
    "Needs Analysis": [30, 35, 40],
    "Value Proposition": [40, 45, 50],
    "Identifying Decision Makers": [50, 55, 60],
    "Perception Analysis": [60, 65, 70],
    "Proposal/Price Quote": [70, 75, 80],
    "Negotiation/Review": [80, 85, 90],
    "Closed Won": [100],  # Must be exactly 100
    "Closed Lost": [0]    # Must be exactly 0
}

# =============================================================================
# INVALID DATA POOLS
# =============================================================================

# Invalid phone numbers with corruption types
# Format: {corruption_type: [list of invalid phones]}
INVALID_PHONES = {
    # 11 digits - added country code prefix
    "prefix_country_code": [
        "15551234567", "12025551234", "13105559876", "14155550123", "16505554321",
        "17185556789", "18185550987", "19145553456", "16175557890", "13125552345"
    ],
    # 9 digits - one digit short
    "too_few_digits": [
        "555123456", "202555123", "310555987", "415555012", "650555432",
        "718555678", "818555098", "914555345", "617555789", "312555234"
    ],
    # 11 digits - one digit extra
    "too_many_digits": [
        "55512345678", "20255512345", "31055598765", "41555501234", "65055543210",
        "71855567890", "81855509876", "91455534567", "61755578901", "31255523456"
    ]
}

# Invalid email domains (test domains that should trigger validation)
INVALID_EMAIL_DOMAINS = {
    "test_domain": ["test"],
    "example_domain": ["example"],
    "invalid_domain": ["invalid"],
    "localhost_domain": ["localhost"]
}

# Invalid US state abbreviations (20 two-letter combinations, no duplicates)
# None of these match any valid US state abbreviation
INVALID_STATES = [
    "AB", "BC", "CD", "DF", "EG", "FH", "GJ", "HK", "JL", "KM",
    "LN", "MP", "NQ", "PR", "QS", "RT", "SU", "TV", "UW", "VX"
]

# Invalid probabilities for closed stages
INVALID_STAGE_PROBABILITIES = {
    # Closed Won should be 100, but these are not (all below 80)
    "closed_won_not_100": [10, 20, 30, 40, 50, 60, 70],
    # Closed Lost should be 0, but these are not
    "closed_lost_not_0": [5, 10, 15, 20, 25, 30, 40, 50, 60, 70]
}


# =============================================================================
# GENERATOR FUNCTIONS
# =============================================================================

def get_phone(force_valid: Optional[bool] = None) -> Tuple[str, bool, Optional[str]]:
    """
    Generate a phone number with configurable validity.
    
    Args:
        force_valid: If True, always return valid. If False, always return invalid.
                    If None, use the global invalid probability.
    
    Returns:
        Tuple of (phone_number, is_valid, corruption_type)
        - phone_number: The generated phone number (plain digits)
        - is_valid: Whether the phone is valid (exactly 10 digits)
        - corruption_type: None if valid, otherwise the type of corruption applied
    """
    # Determine if we should generate valid or invalid
    if force_valid is True:
        should_be_valid = True
    elif force_valid is False:
        should_be_valid = False
    else:
        should_be_valid = random.random() >= _INVALID_FIELD_PROBABILITY
    
    if should_be_valid:
        return (random.choice(VALID_PHONES), True, None)
    else:
        # Choose a random corruption type
        corruption_type = random.choice(list(INVALID_PHONES.keys()))
        phone = random.choice(INVALID_PHONES[corruption_type])
        return (phone, False, corruption_type)


def get_email(first_name: str = None, last_name: str = None, 
              force_valid: Optional[bool] = None) -> Tuple[str, bool, Optional[str]]:
    """
    Generate an email address with configurable validity.
    
    Args:
        first_name: First name for email generation (random if not provided)
        last_name: Last name for email generation (random if not provided)
        force_valid: If True, always return valid. If False, always return invalid.
                    If None, use the global invalid probability.
    
    Returns:
        Tuple of (email, is_valid, corruption_type)
        - email: The generated email address
        - is_valid: Whether the email is valid (uses real domain)
        - corruption_type: None if valid, otherwise the invalid domain type
    """
    # Use random names if not provided
    if first_name is None:
        first_name = random.choice(FIRST_NAMES)
    if last_name is None:
        last_name = random.choice(LAST_NAMES)
    
    # Generate username part
    formats = [
        f"{first_name.lower()}.{last_name.lower()}",
        f"{last_name.lower()}.{first_name.lower()}",
        f"{first_name.lower()}{last_name.lower()}",
        f"{first_name[0].lower()}{last_name.lower()}"
    ]
    username = random.choice(formats)
    
    # Determine if we should generate valid or invalid
    if force_valid is True:
        should_be_valid = True
    elif force_valid is False:
        should_be_valid = False
    else:
        should_be_valid = random.random() >= _INVALID_FIELD_PROBABILITY
    
    if should_be_valid:
        domain = random.choice(VALID_EMAIL_DOMAINS)
        return (f"{username}@{domain}", True, None)
    else:
        # Choose a random invalid domain type
        corruption_type = random.choice(list(INVALID_EMAIL_DOMAINS.keys()))
        invalid_tld = random.choice(INVALID_EMAIL_DOMAINS[corruption_type])
        # Create email with company name + invalid TLD
        company_names = ["acme", "globex", "initech", "umbrella", "stark", "wayne"]
        company = random.choice(company_names)
        return (f"{username}@{company}.{invalid_tld}", False, corruption_type)


def get_state(force_valid: Optional[bool] = None) -> Tuple[str, bool, Optional[str]]:
    """
    Generate a US state abbreviation with configurable validity.
    
    Args:
        force_valid: If True, always return valid. If False, always return invalid.
                    If None, use the global invalid probability.
    
    Returns:
        Tuple of (state, is_valid, corruption_type)
        - state: The generated state abbreviation (2 letters)
        - is_valid: Whether it's a valid US state
        - corruption_type: None if valid, "invalid_abbreviation" if invalid
    """
    # Determine if we should generate valid or invalid
    if force_valid is True:
        should_be_valid = True
    elif force_valid is False:
        should_be_valid = False
    else:
        should_be_valid = random.random() >= _INVALID_FIELD_PROBABILITY
    
    if should_be_valid:
        return (random.choice(VALID_US_STATES), True, None)
    else:
        return (random.choice(INVALID_STATES), False, "invalid_abbreviation")


def get_probability_for_stage(stage: str, force_valid: Optional[bool] = None) -> Tuple[int, bool, Optional[str]]:
    """
    Generate a probability value for a given sales stage with configurable validity.
    
    Only applies validation for "Closed Won" and "Closed Lost" stages.
    Other stages always return valid probabilities.
    
    Args:
        stage: The sales stage name
        force_valid: If True, always return valid. If False, always return invalid
                    (only for Closed Won/Lost). If None, use the global probability.
    
    Returns:
        Tuple of (probability, is_valid, corruption_type)
        - probability: The generated probability (0-100)
        - is_valid: Whether it's consistent with the stage
        - corruption_type: None if valid, otherwise the type of inconsistency
    """
    # Only Closed Won and Closed Lost have strict probability requirements
    if stage == "Closed Won":
        if force_valid is True:
            return (100, True, None)
        elif force_valid is False:
            prob = random.choice(INVALID_STAGE_PROBABILITIES["closed_won_not_100"])
            return (prob, False, "closed_won_not_100")
        else:
            if random.random() >= _INVALID_FIELD_PROBABILITY:
                return (100, True, None)
            else:
                prob = random.choice(INVALID_STAGE_PROBABILITIES["closed_won_not_100"])
                return (prob, False, "closed_won_not_100")
    
    elif stage == "Closed Lost":
        if force_valid is True:
            return (0, True, None)
        elif force_valid is False:
            prob = random.choice(INVALID_STAGE_PROBABILITIES["closed_lost_not_0"])
            return (prob, False, "closed_lost_not_0")
        else:
            if random.random() >= _INVALID_FIELD_PROBABILITY:
                return (0, True, None)
            else:
                prob = random.choice(INVALID_STAGE_PROBABILITIES["closed_lost_not_0"])
                return (prob, False, "closed_lost_not_0")
    
    else:
        # For other stages, return a valid probability from the range
        if stage in VALID_STAGE_PROBABILITIES:
            prob = random.choice(VALID_STAGE_PROBABILITIES[stage])
        else:
            prob = random.randint(10, 90)
        return (prob, True, None)


def get_full_name() -> Tuple[str, str, str]:
    """
    Generate a random full name.
    
    Returns:
        Tuple of (full_name, first_name, last_name)
    """
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return (f"{first} {last}", first, last)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_validation_metadata(fields: dict) -> dict:
    """
    Generate validation metadata for a set of fields.
    
    Args:
        fields: Dictionary of field_name -> (value, is_valid, corruption_type)
    
    Returns:
        Dictionary suitable for including in task output
    """
    metadata = {}
    for field_name, (value, is_valid, corruption) in fields.items():
        metadata[field_name] = {
            "value": value,
            "is_valid": is_valid,
            "corruption": corruption
        }
    return metadata


# =============================================================================
# ADDITIONAL DATA POOLS (from original generate_augmented_tasks.py)
# =============================================================================

COMPANY_SUFFIXES = [
    "Inc.", "LLC", "Corp.", "Ltd.", "Industries", "Solutions", "Group", "Enterprises",
    "Systems", "Technologies", "Consulting", "Services", "Partners", "Associates"
]

JOB_TITLES = [
    "Marketing Director", "Sales Manager", "Product Designer", "Software Engineer",
    "Operations Manager", "HR Director", "Finance Manager", "Business Analyst",
    "Project Manager", "Account Executive", "Customer Success Manager", "Data Analyst"
]

DEPARTMENTS = [
    "Marketing", "Sales", "Engineering", "Operations", "HR", "Finance",
    "Product", "Customer Success", "Business Development", "Legal"
]

EMAIL_STATUSES = ["Primary", "Opt Out", "Invalid"]

PRIORITIES = ["High", "Medium", "Low"]

SALES_STAGES = [
    "Prospecting", "Qualification", "Needs Analysis", "Value Proposition",
    "Identifying Decision Makers", "Perception Analysis", "Proposal/Price Quote",
    "Negotiation/Review", "Closed Won", "Closed Lost"
]

TASK_STATUSES = [
    "Not Started", "In Progress", "Completed", "Pending Input", "Deferred"
]

# Predefined entities for UPDATE tasks (demo data)
PREDEFINED_CONTACTS = [
    "Pam Beesly", "Toby Flenderson", "Jim Halpert", "Ryan Howard",
    "Stanley Hudson", "Kevin Malone", "Angela Martin", "Oscar Martinez",
    "Dwight Schrute", "Michael Scott"
]

PREDEFINED_ACCOUNTS = [
    "Acme Corporation", "Globex Industries", "Soylent Corp", "Initech",
    "Umbrella Corporation", "Massive Dynamic", "Stark Industries",
    "Wayne Enterprises", "Wonka Industries"
]

PREDEFINED_OPPORTUNITIES = [
    "Website Redesign", "Mobile App Development", "Cloud Migration",
    "Cybersecurity Upgrade", "Data Analytics Implementation", "AI Integration",
    "ERP Deployment", "Marketing Automation", "Customer Portal Setup",
    "Inventory Management", "Marketing Campaign"
]

PREDEFINED_LEADS = [
    "Bruce Wayne", "Clark Kent", "Diana Prince", "Barry Allen",
    "Hal Jordan", "Arthur Curry", "Victor Stone", "Peter Parker",
    "Tony Stark", "Natasha Romanoff"
]


if __name__ == "__main__":
    # Demo/test the generators
    print("=" * 60)
    print("Field Data Pools - Demo")
    print("=" * 60)
    print(f"\nCurrent invalid probability: {get_invalid_probability() * 100:.0f}%\n")
    
    print("Phone Numbers:")
    for i in range(5):
        phone, valid, corruption = get_phone()
        status = "VALID" if valid else f"INVALID ({corruption})"
        print(f"  {phone} - {status}")
    
    print("\nEmail Addresses:")
    for i in range(5):
        email, valid, corruption = get_email()
        status = "VALID" if valid else f"INVALID ({corruption})"
        print(f"  {email} - {status}")
    
    print("\nUS States:")
    for i in range(5):
        state, valid, corruption = get_state()
        status = "VALID" if valid else f"INVALID ({corruption})"
        print(f"  {state} - {status}")
    
    print("\nProbabilities for Closed Won:")
    for i in range(5):
        prob, valid, corruption = get_probability_for_stage("Closed Won")
        status = "VALID" if valid else f"INVALID ({corruption})"
        print(f"  {prob}% - {status}")
    
    print("\nProbabilities for Closed Lost:")
    for i in range(5):
        prob, valid, corruption = get_probability_for_stage("Closed Lost")
        status = "VALID" if valid else f"INVALID ({corruption})"
        print(f"  {prob}% - {status}")
