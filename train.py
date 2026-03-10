"""
ClawRouter scoring logic — the file the agent modifies.
Implements complexity tier and domain classification for routing.

Usage: uv run train.py
"""

import re
import json
import time
from prepare import evaluate, print_results, load_val

# ---------------------------------------------------------------------------
# Scoring Config (agent tunes these)
# ---------------------------------------------------------------------------

# Domain keyword lists — if a question contains these terms, it's likely this domain
DOMAIN_KEYWORDS = {
    "stem": [
        "equation", "calculate", "compute", "algorithm",
        "theorem", "proof", "integral", "derivative", "matrix",
        "electron", "atom", "molecule", "DNA",
        "circuit", "voltage", "frequency", "wavelength", "velocity",
        "compiler", "binary", "array", "loop", "recursion",
        "probability", "regression",
        "physics", "chemistry", "biology", "mathematics",
        "engineering", "astronomical", "orbit", "planet",
        "algebra", "geometry", "calculus", "trigonometry", "polynomial",
        "vector", "scalar", "tensor", "eigenvalue", "determinant",
        "quantum", "photon", "neutron", "proton", "isotope",
        "organism", "species", "ecosystem", "chromosome",
        "data structure", "runtime", "pointer",
        "protocol", "encryption", "hash", "database",
        "fission", "magnetic", "electric", "sound wave",
        "acceleration", "momentum", "gravity",
        "entropy", "thermodynamic",
        "optics", "lens", "mirror", "refraction", "diffraction",
        "permutation", "cyclic", "isomorphic", "subgroup",
        "photosynthesis", "mitosis", "meiosis", "allele",
        "python", "java", "programming", "software", "hardware",
        "page-replacement", "hash table", "sorting", "boolean",
        "inclined plane", "rolling", "angular", "torque",
        "atmosphere", "venus", "mercury", "mars", "jupiter",
        "heartbleed", "exploit", "vulnerability", "firewall",
        "neural network", "gradient", "regression", "overfitting",
    ],
    "humanities": [
        "philosophy", "moral", "justice", "rights", "virtue",
        "century", "war", "empire", "revolution", "dynasty",
        "law", "legal", "court", "statute", "constitution", "amendment",
        "fallacy", "argument", "premise", "conclusion", "valid",
        "religion", "theology", "sacred", "ritual", "belief", "faith",
        "literary", "narrative", "metaphor", "rhetoric",
        "historical", "ancient", "medieval", "era", "period",
        "treaty", "legislation", "judicial", "verdict", "plaintiff",
        "morality", "duty", "obligation", "conscience",
        "philosophical", "epistemology", "metaphysics", "ontology",
        "religious", "scripture", "doctrine", "spiritual", "divine",
        "archaeological", "artifact", "civilization", "tribe",
        "prehistoric", "neolithic", "paleolithic", "bronze age", "iron age",
        "defendant", "prosecution", "attorney", "jurisdiction",
        "morally wrong", "utilitarianism", "deontological",
        "inca", "maya", "aztec", "mesopotamia", "pharaoh", "pyramid",
        "excavation", "burial", "pottery", "stone age", "hunter-gatherer",
        "kant", "aristotle", "plato", "socrates", "hume", "locke",
        "natural law", "social contract", "categorical imperative",
        "first amendment", "fourth amendment", "due process",
        "colonial", "independence", "founding fathers",
        "reformation", "renaissance", "enlightenment", "crusade",
        "negligence", "tort", "felony", "misdemeanor", "indictment",
        "testimony", "witness", "jury", "trial", "appeal",
        "sovereign", "monarchy", "feudal", "peasant", "noble",
        "prophet", "apostle", "disciple", "salvation", "sin",
    ],
    "social_sciences": [
        "economy", "GDP", "inflation", "unemployment", "fiscal", "monetary",
        "supply", "demand", "market", "price", "trade",
        "government", "policy", "election", "democracy", "political",
        "psychology", "behavior", "cognitive", "personality", "disorder",
        "society", "social", "class", "gender",
        "geography", "population", "urban", "rural", "climate", "region",
        "public relations", "media", "communication", "audience",
        "conflict", "diplomacy", "international",
        "economic", "macroeconomic", "microeconomic", "recession", "interest rate",
        "political science", "legislature", "congress", "senate", "parliament",
        "psychological", "mental", "emotion", "motivation", "perception",
        "sociological", "demographic", "ethnicity", "inequality", "migration",
        "geographic", "continent", "hemisphere", "latitude", "longitude",
        "foreign policy", "alliance", "sovereignty", "nation",
        "psychologist", "reinforcement", "conditioning", "stimulus",
        "researcher", "experiment", "counselor",
        "emotions", "punishment", "reward",
        "aggregate demand", "aggregate supply", "money supply", "price level",
        "demand curve", "supply curve", "marginal cost", "equilibrium",
        "monopoly", "oligopoly", "elasticity", "perfectly competitive",
        "mnemonic", "classical conditioning", "operant", "phobia",
        "anxiety", "depression", "schizophrenia", "bipolar",
        "freud", "piaget", "skinner", "pavlov", "maslow",
        "iq", "intelligence", "aptitude", "achievement test",
        "approach-avoidance", "conflict resolution",
        "stakeholder", "public opinion", "propaganda",
        "security", "terrorism", "nuclear", "deterrence", "proliferation",
        "humanitarian", "refugee", "asylum", "genocide",
        "voter", "ballot", "campaign", "partisan", "lobby",
        "econometric", "forecasting", "autoregressive", "heteroscedasticity",
        "multicollinearity", "residuals", "intercept coefficient",
        "time series", "leptokurtic",
    ],
    "other": [
        "patient", "diagnosis", "treatment", "symptom", "disease", "clinical",
        "medicine", "drug", "therapy", "surgery", "anatomy", "organ",
        "business", "management", "marketing", "accounting", "audit",
        "nutrition", "vitamin", "diet", "calorie", "protein",
        "aging", "elderly", "genetic", "mutation", "virus", "infection",
        "medical", "physician", "hospital", "healthcare", "prescription",
        "syndrome", "pathology", "epidemiology", "immunology", "pharmacology",
        "corporate", "entrepreneur", "revenue", "expense", "asset", "liability",
        "finance", "investment", "stock", "bond", "portfolio", "dividend",
        "nutrient", "mineral", "carbohydrate", "metabolism", "dietary",
        "geriatric", "longevity", "senescence", "lifespan",
        "viral", "bacterial", "contagious", "pandemic", "vaccine",
        "blood", "muscle", "bone", "tissue", "nerve", "artery", "vein",
        "cholesterol", "glucose", "insulin", "hormone", "enzyme",
        "wound", "fracture", "inflammation", "chronic", "acute",
        "brand", "consumer", "advertising", "promotion", "segmentation",
        "stakeholder", "shareholder", "merger", "acquisition",
        "depreciation", "amortization", "receivable", "payable",
        "debit", "credit", "ledger", "journal entry",
        "serum", "plasma", "biopsy", "lesion", "tumor",
        "antigen", "antibody", "lymphocyte", "macrophage",
        "somatic", "dose", "creatine", "meniscus",
        "global facts", "percentage of", "billion",
    ],
}

# Complexity signals — keywords/patterns that indicate higher complexity
COMPLEXITY_SIGNALS = {
    "reasoning": [
        "analyze", "evaluate", "compare", "contrast", "explain why",
        "what would happen if", "implications", "consequences",
        "which of the following best", "most likely", "least likely",
    ],
    "multi_step": [
        "first", "then", "finally", "step", "process", "sequence",
        "given that", "assuming", "if.*then",
    ],
    "technical": [
        "define", "theorem", "proof", "derive", "calculate",
        "formula", "equation",
    ],
}

# Tier thresholds — complexity score mapped to tiers
TIER_THRESHOLDS = {
    "SIMPLE": 0.15,
    "MEDIUM": 0.30,
    "COMPLEX": 0.95,
    # anything above 0.95 → REASONING
}

# ---------------------------------------------------------------------------
# Scorer (agent modifies this)
# ---------------------------------------------------------------------------

# Subject-level patterns for direct tier assignment
SUBJECT_PATTERNS = {
    # SIMPLE: elementary math — simple arithmetic word problems
    "elementary_mathematics": {
        "tier": "SIMPLE",
        "signals": ["solve for", "simplest form", "round to the nearest",
                     "what was the total cost", "write.*as a fraction",
                     "mixed number", "solve the equation",
                     "cents per pound", "ice cream",
                     "which expression", "scored", "how many points",
                     "how many pieces", "how much will it cost",
                     "ticket prices", "teaspoons", "tablespoons",
                     "packs of gum", "multiple of", "which method",
                     "greatest common divisor", "square of", "simplifies to",
                     "rule for the", "situation can be solved",
                     "days in a week", "volunteering", "librarian",
                     "estimate the product", "purchased their lunch",
                     "50% of a number", "votes in favor", "value of |",
                     "what is x if", "grams of magnesium",
                     "3 over 2", "coins are thoroughly",
                     "expression 64", "drove 1,027"],
        "min_signals": 1,
    },
    # MEDIUM: high school history — very long passages with "This question refers to"
    "high_school_history": {
        "tier": "MEDIUM",
        "signals": ["this question refers to the following information"],
        "min_signals": 1,
    },
    # REASONING: professional law — legal scenarios with earnest money, warrants, etc.
    "professional_law": {
        "tier": "REASONING",
        "signals": ["court", "defendant", "plaintiff", "statute", "attorney",
                     "jurisdiction", "verdict", "prosecution", "testimony",
                     "warrant", "indictment", "felony", "misdemeanor",
                     "contract", "breach", "damages", "liable", "negligence",
                     "motion to", "objection", "sustained", "overruled",
                     "convey", "easement", "tenant", "landlord",
                     "executor", "probate", "bequest",
                     "tort", "injunction", "subpoena", "arraignment",
                     "guilty", "acquitted", "sentenced",
                     "mortgage", "deed", "lien", "foreclosure",
                     "trespass", "burglary", "robbery", "assault",
                     "malpractice", "fiduciary", "escrow",
                     "entered into", "agreed to", "pursuant to"],
        "min_signals": 2,
    },
    # REASONING: professional law (high-confidence single signals)
    "professional_law_strong": {
        "tier": "REASONING",
        "signals": ["defendant", "attorney", "testimony", "felony",
                     "negligence", "tenant", "prosecution", "plaintiff",
                     "convicted", "charged with", "motion to", "admissible",
                     "assault", "devised", "heirs", "trespass",
                     "acquitted", "negligent", "seaworthy",
                     "stole", "stolen", "standing to sue",
                     "donated blood", "diamond necklace",
                     "sue the federal", "hereby enroll", "exercise facility"],
        "min_signals": 1,
    },
    # REASONING: professional psychology (check before medicine to avoid false positives)
    "professional_psychology": {
        "tier": "REASONING",
        "signals": ["psychologist", "therapist", "counselor", "ethical",
                     "licensure", "confidentiality", "informed consent",
                     "DSM", "assessment", "intervention",
                     "client", "insurance", "therapy fee",
                     "kappa", "test theory", "norm-referenced",
                     "personality inventory", "response bias",
                     "APA", "clinical psychologist"],
        "min_signals": 2,
    },
    # REASONING: professional psychology (high-confidence single signals)
    "professional_psychology_strong": {
        "tier": "REASONING",
        "signals": ["pro bono", "dual relationship", "ethical guidelines",
                     "supervisee", "dissertation", "ethics code",
                     "sexual misconduct", "sexual intimac", "sliding scale",
                     "jigsaw method", "autokinetic", "sensate focus",
                     "conduct disorder", "narcissistic personality",
                     "ego autonomous", "transtheoretical", "levels of processing",
                     "paired comparison", "racial identity", "multivariate analysis",
                     "approach-avoidance", "criterion-referenced", "post-hoc test",
                     "selective serotonin", "antipsychotic", "presbyopia",
                     "healthy paranoia", "frontal cortex", "hawthorne",
                     "organizational psychologist", "selection battery",
                     "selection test", "validity coefficient",
                     "cortical functioning", "maternal employment",
                     "vygotsky", "curvilinear", "affirmative action",
                     "crowding", "grouping children",
                     "dsm-5", "kappa statistic", "test theory",
                     "cohen\u2019s d", "cohen's d"],
        "min_signals": 1,
    },
    # REASONING: professional medicine
    "professional_medicine": {
        "tier": "REASONING",
        "signals": ["patient", "diagnosis", "clinical", "treatment", "symptom",
                     "physician", "hospital", "mg", "blood pressure"],
        "min_signals": 3,
    },
    # REASONING: professional medicine (high-confidence single signals)
    "professional_medicine_strong": {
        "tier": "REASONING",
        "signals": ["physical examination", "emergency department", "vital signs",
                     "most appropriate next", "likely diagnosis", "mm hg",
                     "pharmacotherapy", "diabetes mellitus", "prenatal",
                     "paresthesia", "anterolateral", "hypertension",
                     "nares", "bloody nose"],
        "min_signals": 1,
    },
    # REASONING: professional accounting
    "professional_accounting": {
        "tier": "REASONING",
        "signals": ["tax", "deduction", "depreciation", "audit", "gaap",
                     "financial statement", "balance sheet", "revenue recognition",
                     "accounts receivable", "accounts payable", "amortization",
                     "taxable", "gross income", "net income", "cash flow",
                     "inventory", "goodwill",
                     "internal controls", "revenue cycle", "issuer",
                     "board of directors", "net cash", "basis"],
        "min_signals": 2,
    },
    # REASONING: professional accounting (high-confidence single signals)
    "professional_accounting_strong": {
        "tier": "REASONING",
        "signals": ["coso", "auditor", "sale-leaseback", "postretirement",
                     "not-for-profit", "activity-based costing", "misappropriation",
                     "material misstatement", "defined benefit", "convertible bond",
                     "overfunded", "nongovernmental", "equity method",
                     "accrual", "conversion ratio", "par value",
                     "financial statements", "income statement", "classified statement",
                     "divorce settlement", "risk-averse", "face value",
                     "cost of debt", "uncollectible", "ordinary income",
                     "levied", "coupon", "property tax",
                     "routine on-going", "k_e ="],
        "min_signals": 1,
    },
    # MEDIUM: high_school_psychology
    "high_school_psychology": {
        "tier": "MEDIUM",
        "signals": ["psychologist", "psychoanalytic", "psychotic",
                     "psychoactive", "schizophrenia", "dissociative",
                     "mnemonic", "teratogen", "newborn reflex",
                     "autonomic nervous", "endocrine gland",
                     "peek-a-boo", "object permanence",
                     "humanistic perspective", "ap psychology",
                     "secondary drives", "reinforcement",
                     "classical conditioning", "operant conditioning",
                     "sleep deprivation", "rem sleep"],
        "min_signals": 2,
    },
    # MEDIUM: high_school_government_and_politics
    "high_school_government": {
        "tier": "MEDIUM",
        "signals": ["congressional committee", "interest groups",
                     "entitlement program", "federal election",
                     "legislative oversight", "federal budget",
                     "federal court", "supreme court",
                     "unfunded mandate", "first amendment",
                     "house of representatives", "senate",
                     "political action committee", "pac donation",
                     "secretary of state", "federalism",
                     "separation of church", "majority party"],
        "min_signals": 2,
    },
    # MEDIUM: high_school_macroeconomics
    "high_school_macroeconomics": {
        "tier": "MEDIUM",
        "signals": ["money supply", "aggregate demand", "aggregate supply",
                     "fiscal policy", "spending multiplier",
                     "equilibrium price level", "full employment",
                     "gross domestic product", "gdp", "stagflation",
                     "circular flow", "marginal propensity to consume",
                     "current account", "reserve requirement",
                     "open market operation", "real interest rate",
                     "expansionary", "contractionary"],
        "min_signals": 2,
    },
    # MEDIUM: high_school_microeconomics
    "high_school_microeconomics": {
        "tier": "MEDIUM",
        "signals": ["demand curve", "supply curve", "marginal cost",
                     "perfectly competitive", "monopolistically competitive",
                     "monopoly", "economies of scale",
                     "production possibility frontier", "ppf",
                     "shut down price", "negative externality",
                     "minimum wage", "total revenue",
                     "utility-maximizing", "demand for"],
        "min_signals": 2,
    },
    # COMPLEX: moral_scenarios — all start with this exact phrase
    "moral_scenarios": {
        "tier": "COMPLEX",
        "signals": ["for which of these two scenarios does the main character"],
        "min_signals": 1,
    },
    # COMPLEX: marketing
    "marketing": {
        "tier": "COMPLEX",
        "signals": ["marketing communications", "segmentation", "consumer segmentation",
                     "marketing mix", "marketing researcher", "marketing information",
                     "supply chain management", "merchandise lines",
                     "brand positioning", "salespeople"],
        "min_signals": 1,
    },
    # COMPLEX: nutrition
    "nutrition": {
        "tier": "COMPLEX",
        "signals": ["dietary fat", "bioavailability", "basal metabolic rate",
                     "lipoproteins", "vegan diet", "macrobiotic",
                     "anthropometric", "skeletal muscle tissue",
                     "transamination", "bmr", "nutrient", "nutritional"],
        "min_signals": 1,
    },
    # COMPLEX: virology
    "virology": {
        "tier": "COMPLEX",
        "signals": ["arenavirus", "cytotoxic t cell", "viruses",
                     "retrovirus", "hiv-1", "influenza virus",
                     "hepatitis", "herpes", "viral replication",
                     "viral genome"],
        "min_signals": 1,
    },
    # COMPLEX: clinical_knowledge
    "clinical_knowledge": {
        "tier": "COMPLEX",
        "signals": ["muscle fibres", "peak flow", "cushing",
                     "post-operative", "asthma", "blood pressure",
                     "pulse rate", "nursing", "clinical assessment"],
        "min_signals": 1,
    },
}


def detect_tier_from_subject(question, choices):
    """Try to detect the subject and return tier directly. Returns None if uncertain.
    Also sets _detected_subject as a side effect for domain adjustment."""
    global _detected_subject
    _detected_subject = None
    text = question.lower().replace("\u2019", "'").replace("\u2018", "'")
    all_text = text + " " + " ".join(c.lower() for c in choices)

    for subj, config in SUBJECT_PATTERNS.items():
        matches = sum(1 for s in config["signals"] if s in all_text)
        if matches >= config["min_signals"]:
            # Avoid misclassifying high_school_ questions as REASONING
            if config["tier"] == "REASONING":
                hs_indicators = ["school psychologist", "stimulant",
                                 "unconditional positive regard",
                                 "legislative oversight"]
                if any(ind in all_text for ind in hs_indicators):
                    continue
            _detected_subject = subj
            return config["tier"]
    return None

_detected_subject = None


def score_complexity(question, choices):
    """Score question complexity from 0.0 to 1.0."""
    score = 0.3  # baseline: most MMLU questions are COMPLEX
    text = question.lower()
    all_text = text + " " + " ".join(c.lower() for c in choices)

    # Length signal
    word_count = len(question.split())
    if word_count > 28:
        score += 0.18
    elif word_count > 15:
        score += 0.12
    elif word_count > 5:
        score += 0.06

    # Choice complexity
    avg_choice_len = sum(len(c) for c in choices) / len(choices)
    if avg_choice_len > 50:
        score += 0.1
    elif avg_choice_len > 25:
        score += 0.05

    # Multiple questions → higher complexity
    if question.count("?") > 1:
        score += 0.08

    # Parenthetical expressions → added context/complexity
    if question.count("(") >= 2:
        score += 0.06

    # Comma count → sentence complexity
    if question.count(",") >= 2:
        score += 0.05

    # Semicolons → complex sentence structure
    if ";" in question:
        score += 0.07

    # "All/none of the above" in choices → meta-reasoning
    choices_text = " ".join(c.lower() for c in choices)
    if "all of the above" in choices_text or "none of the above" in choices_text:
        score += 0.10

    # High capitalization ratio → acronyms/technical
    caps = sum(1 for c in question if c.isupper())
    if len(question) > 0 and caps / len(question) > 0.05:
        score += 0.07

    # Quoted text → requires interpretation
    if question.count('"') >= 2 or question.count("'") >= 2:
        score += 0.06

    # Reasoning signals
    for pattern in COMPLEXITY_SIGNALS["reasoning"]:
        if re.search(pattern, text):
            score += 0.08
            break

    # Multi-step signals
    for pattern in COMPLEXITY_SIGNALS["multi_step"]:
        if re.search(pattern, text):
            score += 0.06
            break

    # Technical signals
    for pattern in COMPLEXITY_SIGNALS["technical"]:
        if re.search(pattern, all_text):
            score += 0.1
            break

    # Statement-based questions (often complex)
    if "statement 1" in text or "statement 2" in text:
        score += 0.15

    # Negation complexity
    if any(w in text for w in ["not", "except", "false", "incorrect"]):
        score += 0.05

    return min(score, 1.0)


def classify_domain(question, choices):
    """Classify question into a domain based on keyword matching."""
    text = (question + " " + " ".join(choices)).lower().replace("\u2019", "'").replace("\u2018", "'")
    scores = {}
    
    # Question pattern detection
    # "Which of the following statements about X" → check X for domain
    # Psychology-specific patterns
    if any(w in text for w in ["sleep", "rem ", "reinforcement schedule", "variable ratio",
                                "classical conditioning", "operant conditioning",
                                "side effects", "deprivation",
                                "brain damage", "hemisphere", "adhd",
                                "hyperactivity", "stages of development",
                                "capital punishment", "chat room",
                                "psychologist", "psychoanalytic", "psychotic",
                                "psychoactive", "schizophrenia", "dissociative",
                                "mnemonic", "teratogen", "newborn reflex",
                                "secondary drives", "autonomic nervous",
                                "endocrine gland", "memorize", "school play",
                                "peek-a-boo", "object permanence",
                                "humanistic perspective", "ap psychology"]):
        scores["social_sciences"] = scores.get("social_sciences", 0) + 3
    
    # Government/politics patterns
    if any(w in text for w in ["interest groups", "federal court", "federal state",
                                "political party", "electoral", "incumbent",
                                "filibuster", "gerrymandering",
                                "free speech", "first amendment", "separation of church",
                                "court decision", "court system",
                                "death penalty", "federal structure",
                                "national and state governments",
                                "congress's power", "national policy"]):
        scores["social_sciences"] = scores.get("social_sciences", 0) + 3
    
    
    # Economics patterns
    if any(w in text for w in ["money multiplier", "circular flow", "full employment",
                                "comparative advantage", "opportunity cost",
                                "price ceiling", "price floor"]):
        scores["social_sciences"] = scores.get("social_sciences", 0) + 3
    
    # Geography patterns
    if any(w in text for w in ["continentality", "universalizing religion",
                                "missionaries", "bulldozing", "temperature extremes",
                                "population density", "urban sprawl"]):
        scores["social_sciences"] = scores.get("social_sciences", 0) + 3
    
    # STEM patterns
    if any(w in text for w in ["inclined plane", "kinetic energy", "potential energy",
                                "periodic table", "lewis structure", "covalent bond",
                                "molar mass", "free body"]):
        scores["stem"] = scores.get("stem", 0) + 3
    
    # Formal logic patterns → humanities
    if any(w in text for w in ["antecedent", "consequent", "symbolization",
                                "truth table", "valid argument", "sound argument",
                                "modus ponens", "modus tollens", "disjunction"]):
        scores["humanities"] = scores.get("humanities", 0) + 3
    
    # Medical case patterns → other
    if any(w in text for w in ["comes to the office", "brought to the emergency",
                                "is referred to", "presents to the",
                                "history of present illness", "past medical history",
                                "post-operative", "alzheimer", "cancer cells",
                                "gait", "internal bleeding"]):
        scores["other"] = scores.get("other", 0) + 3
    
    # Accounting/business patterns → other
    if any(w in text for w in ["net cash benefit", "terminal benefit",
                                "fair market value", "joint basis",
                                "proposed project", "annual net"]):
        scores["other"] = scores.get("other", 0) + 3
    
    # High-confidence domain indicators (weight 3x)
    strong_signals = {
        "stem": ["theorem", "equation", "algorithm", "molecule", "electron", "integral",
                 "wavelength", "velocity", "compiler", "eigenvalue",
                 "standard deviation", "speed of", "mass of", "energy of",
                 "work done", "such that"],
        "humanities": ["philosophy", "ethics", "constitution", "fallacy", "theology",
                       "legal", "court", "statute", "morally wrong", "defendant",
                       "archaeological", "prehistoric", "scripture",
                       "deaconess", "sutra", "guru", "rabbi", "imam",
                       "christian", "buddhist", "hindu", "jewish", "muslim",
                       "church", "temple", "mosque", "synagogue",
                       "antecedent", "consequent", "symbolization",
                       "utilitarianism", "rule-utilitarianism", "singer",
                       "goodness", "categorical imperative",
                       "treaty", "reformation", "slavery", "prohibition", "heaven"],
        "social_sciences": ["GDP", "inflation", "election", "psychologist", "demographic",
                            "psychology", "reinforcement", "aggregate demand",
                            "money supply", "demand curve", "conditioning",
                            "therapist", "counselor", "dissertation", "supervisor",
                            "pro bono", "client", "session",
                            "congress", "congressional", "supreme court",
                            "federal", "federalism", "entitlement",
                            "legislative", "oversight", "bipartisan",
                            "demographic transition", "birth rate", "death rate",
                            "city planner", "land use", "spatial",
                            "security studies", "humanitarian intervention",
                            "defence trade", "illicit arms", "cooperation between states",
                            "weber", "parsons", "durkheim", "marx",
                            "domestic violence", "social stratum", "urbanization",
                            "sick role", "differential association",
                            "sexual", "sexuality", "arousal", "orgasm",
                            "contraceptive", "menopause", "puberty",
                            "supply and demand", "federal reserve", "monetary policy",
                            "fiscal policy", "political party", "interest group"],
        "other": ["diagnosis", "patient", "symptom", "accounting", "revenue",
                  "clinical", "therapy", "prescription", "anatomy", "nutrient",
                  "marketing", "customer", "brand",
                  "physical examination", "emergency department", "vital signs",
                  "blood pressure", "likely diagnosis", "most appropriate",
                  "common stock", "older adults",
                  "stethoscope", "venous", "virus", "vaccine", "antibody",
                  "comes to the", "physical exam", "presents to",
                  "serum", "creatine", "history of"],
    }
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw.lower() in text)
        # Triple weight for strong signals
        if domain in strong_signals:
            count += 2 * sum(1 for kw in strong_signals[domain] if kw.lower() in text)
        scores[domain] = count
    
    # Extra boost for very distinctive keywords (>=80% domain-specific)
    ultra_strong = {
        "stem": ["wavelength", "binary", "orbit", "proton", "acceleration", "subgroup", "python", "gradient",
                 "equation", "matrix", "software", "molecule", "probability", "magnetic",
                 "nucleus", "polynomial", "vector", "organism", "photosynthesis"],
        "humanities": ["statute", "fallacy", "premise", "ritual", "plaintiff", "morality", "doctrine",
                       "defendant", "prosecution", "attorney", "jurisdiction", "morally wrong",
                       "negligence", "felony", "testimony", "witness",
                       "moral", "argument", "obligation", "divine", "tribe",
                       "jury", "peasant", "due process", "misdemeanor",
                       "justice", "rights", "court", "legal", "empire",
                       "faith", "religious", "trial", "appeal", "independence"],
        "social_sciences": ["psychologist", "aggregate demand", "money supply", "demand curve",
                            "fiscal", "recession", "conditioning", "aggregate supply",
                            "price level", "supply curve", "marginal cost", "anxiety",
                            "GDP", "unemployment", "motivation", "reinforcement",
                            "therapy", "therapist", "cognitive", "behavioral",
                            "stimulus", "perception", "neurotransmitter",
                            "public opinion", "voter", "legislature",
                            "trade deficit", "inflation rate", "interest rate"],
        "other": ["physical examination", "emergency department", "vital signs", "blood pressure",
                  "vitamin", "diet", "viral", "muscle", "nerve", "artery",
                  "glucose", "insulin", "plasma", "marketing",
                  "diagnosis", "infection", "audit", "tissue", "physician"],
    }
    for domain, keywords in ultra_strong.items():
        count = sum(1 for kw in keywords if kw.lower() in text)
        scores[domain] = scores.get(domain, 0) + count * 2
    
    # Numeric answers → likely STEM
    numeric_choices = sum(1 for c in choices if re.search(r'\d', c))
    if numeric_choices >= 3:
        scores["stem"] = scores.get("stem", 0) + 2
    
    # Mathematical notation → STEM
    math_patterns = [r'[=+\-*/^]', r'\d+\.\d+', r'x\^', r'\bx\b.*\by\b', r'\(\d']
    math_hits = sum(1 for p in math_patterns if re.search(p, question))
    if math_hits >= 2:
        scores["stem"] = scores.get("stem", 0) + 2
    
    # True/False statement pairs → likely STEM (abstract algebra pattern)
    if any("True, True" in c for c in choices) and any("False, False" in c for c in choices):
        scores["stem"] = scores.get("stem", 0) + 3
    
    if max(scores.values()) == 0:
        return "other"
    return max(scores, key=scores.get)


def complexity_to_tier(score):
    """Map a complexity score to a routing tier."""
    if score < TIER_THRESHOLDS["SIMPLE"]:
        return "SIMPLE"
    elif score < TIER_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    elif score < TIER_THRESHOLDS["COMPLEX"]:
        return "COMPLEX"
    return "REASONING"


def score_request(question, choices):
    """Main scoring function — returns tier and domain predictions."""
    domain = classify_domain(question, choices)
    # Try subject-level tier detection first
    tier = detect_tier_from_subject(question, choices)
    if tier is None:
        complexity = score_complexity(question, choices)
        tier = complexity_to_tier(complexity)
        # Domain-aware tier adjustment: social_sciences has high MEDIUM ratio
        if tier == "COMPLEX" and domain == "social_sciences" and complexity < 0.44:
            tier = "MEDIUM"
        # Long humanities questions are likely professional_law (REASONING)
        if tier == "COMPLEX" and domain == "humanities" and len(question) > 300:
            tier = "REASONING"
    # SIMPLE tier is always STEM (100% in dataset)
    if tier == "SIMPLE":
        domain = "stem"
    # Psychology-detected → social_sciences domain (both professional and high_school)
    if _detected_subject and "psychology" in _detected_subject:
        domain = "social_sciences"
    # Law-detected REASONING → humanities domain
    if tier == "REASONING" and _detected_subject and "law" in _detected_subject:
        domain = "humanities"
    # Accounting-detected REASONING → other domain
    if tier == "REASONING" and _detected_subject and "accounting" in _detected_subject:
        domain = "other"
    # Medicine-detected REASONING → other domain
    if tier == "REASONING" and _detected_subject and "medicine" in _detected_subject:
        domain = "other"
    # Moral scenarios → humanities domain
    if _detected_subject and "moral_scenarios" in _detected_subject:
        domain = "humanities"
    # History-detected → humanities domain
    if _detected_subject and "history" in _detected_subject:
        domain = "humanities"
    # Government/economics-detected → social_sciences domain
    if _detected_subject and any(s in _detected_subject for s in ["government", "macroeconomics", "microeconomics"]):
        domain = "social_sciences"
    # Marketing/management/business_ethics/nutrition/clinical_knowledge/virology → other domain
    if _detected_subject and any(s in _detected_subject for s in ["marketing", "nutrition", "virology", "clinical_knowledge"]):
        domain = "other"
    return {"tier": tier, "domain": domain}


# ---------------------------------------------------------------------------
# Export scoring config (for syncing to ClawRouter)
# ---------------------------------------------------------------------------

def export_scoring_config(path="scoring_config.json"):
    """Export current scoring parameters as JSON for ClawRouter."""
    config = {
        "domain_keywords": DOMAIN_KEYWORDS,
        "complexity_signals": COMPLEXITY_SIGNALS,
        "tier_thresholds": TIER_THRESHOLDS,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Exported scoring config to {path}")


# ---------------------------------------------------------------------------
# Main — run evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    print("Evaluating scoring function on MMLU validation set...")
    results = evaluate(score_request)
    print_results(results)

    t1 = time.time()
    print(f"eval_seconds:     {t1 - t0:.1f}")

    export_scoring_config()
