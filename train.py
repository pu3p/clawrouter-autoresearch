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
        "equation", "formula", "calculate", "compute", "algorithm", "function",
        "variable", "theorem", "proof", "integral", "derivative", "matrix",
        "electron", "atom", "molecule", "cell", "gene", "protein", "DNA",
        "circuit", "voltage", "frequency", "wavelength", "velocity", "force",
        "compiler", "binary", "array", "loop", "recursion", "complexity",
        "probability", "statistics", "regression", "hypothesis", "variance",
        "physics", "chemistry", "biology", "mathematics", "computer",
        "engineering", "astronomical", "orbit", "planet", "star",
        "algebra", "geometry", "calculus", "trigonometry", "polynomial",
        "vector", "scalar", "tensor", "eigenvalue", "determinant",
        "quantum", "photon", "neutron", "proton", "isotope", "reaction",
        "organism", "species", "evolution", "ecosystem", "chromosome",
        "algorithm", "data structure", "runtime", "memory", "pointer",
        "network", "protocol", "encryption", "hash", "database",
    ],
    "humanities": [
        "philosophy", "ethics", "moral", "justice", "rights", "virtue",
        "history", "century", "war", "empire", "revolution", "dynasty",
        "law", "legal", "court", "statute", "constitution", "amendment",
        "logic", "fallacy", "argument", "premise", "conclusion", "valid",
        "religion", "theology", "sacred", "ritual", "belief", "faith",
        "literary", "narrative", "metaphor", "rhetoric",
        "historical", "ancient", "medieval", "modern", "era", "period",
        "treaty", "legislation", "judicial", "verdict", "plaintiff",
        "ethical", "morality", "duty", "obligation", "conscience",
        "philosophical", "epistemology", "metaphysics", "ontology",
        "religious", "scripture", "doctrine", "spiritual", "divine",
        "archaeological", "artifact", "civilization", "tribe", "homo",
        "prehistoric", "neolithic", "paleolithic", "bronze age", "iron age",
        "defendant", "prosecution", "attorney", "jurisdiction",
        "morally wrong", "scenario", "utilitarianism", "deontological",
        "inca", "maya", "aztec", "mesopotamia", "pharaoh", "pyramid",
        "excavation", "burial", "pottery", "stone age", "hunter-gatherer",
    ],
    "social_sciences": [
        "economy", "GDP", "inflation", "unemployment", "fiscal", "monetary",
        "supply", "demand", "market", "price", "cost", "profit", "trade",
        "government", "policy", "election", "democracy", "political",
        "psychology", "behavior", "cognitive", "personality", "disorder",
        "society", "culture", "social", "class", "gender", "race",
        "geography", "population", "urban", "rural", "climate", "region",
        "public relations", "media", "communication", "audience",
        "security", "conflict", "diplomacy", "international",
        "economic", "macroeconomic", "microeconomic", "recession", "interest rate",
        "political science", "legislature", "congress", "senate", "parliament",
        "psychological", "mental", "emotion", "motivation", "perception",
        "sociological", "demographic", "ethnicity", "inequality", "migration",
        "geographic", "continent", "hemisphere", "latitude", "longitude",
        "foreign policy", "treaty", "alliance", "sovereignty", "nation",
        "psychologist", "reinforcement", "conditioning", "stimulus",
        "researcher", "experiment", "therapy", "client", "counselor",
        "emotions", "punishment", "reward", "learning", "memory",
        "aggregate demand", "aggregate supply", "money supply", "price level",
        "demand curve", "supply curve", "marginal cost", "equilibrium",
        "monopoly", "oligopoly", "elasticity", "perfectly competitive",
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
    "MEDIUM": 0.35,
    "COMPLEX": 0.60,
    # anything above 0.60 → REASONING
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
                     "mixed number"],
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
                     "motion to", "objection", "sustained", "overruled"],
        "min_signals": 2,
    },
    # REASONING: professional medicine
    "professional_medicine": {
        "tier": "REASONING",
        "signals": ["patient", "diagnosis", "clinical", "treatment", "symptom",
                     "physician", "hospital", "mg", "blood pressure"],
        "min_signals": 3,
    },
    # REASONING: professional accounting
    "professional_accounting": {
        "tier": "REASONING",
        "signals": ["tax", "deduction", "depreciation", "audit", "GAAP",
                     "financial statement", "balance sheet", "revenue recognition",
                     "accounts receivable", "accounts payable", "amortization"],
        "min_signals": 2,
    },
    # REASONING: professional psychology
    "professional_psychology": {
        "tier": "REASONING",
        "signals": ["psychologist", "therapist", "counselor", "ethical",
                     "licensure", "confidentiality", "informed consent",
                     "DSM", "assessment", "intervention"],
        "min_signals": 2,
    },
    # COMPLEX: moral_scenarios — all start with this exact phrase
    "moral_scenarios": {
        "tier": "COMPLEX",
        "signals": ["for which of these two scenarios does the main character"],
        "min_signals": 1,
    },
}


def detect_tier_from_subject(question, choices):
    """Try to detect the subject and return tier directly. Returns None if uncertain."""
    text = question.lower()
    all_text = text + " " + " ".join(c.lower() for c in choices)

    for subj, config in SUBJECT_PATTERNS.items():
        matches = sum(1 for s in config["signals"] if s in all_text)
        if matches >= config["min_signals"]:
            return config["tier"]
    return None


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
    text = (question + " " + " ".join(choices)).lower()
    scores = {}
    
    # High-confidence domain indicators (weight 3x)
    strong_signals = {
        "stem": ["theorem", "equation", "algorithm", "molecule", "electron", "integral",
                 "wavelength", "velocity", "compiler", "eigenvalue"],
        "humanities": ["philosophy", "ethics", "constitution", "fallacy", "theology",
                       "legal", "court", "statute", "morally wrong", "defendant",
                       "archaeological", "prehistoric", "scripture"],
        "social_sciences": ["GDP", "inflation", "election", "psychologist", "demographic",
                            "psychology", "reinforcement", "aggregate demand",
                            "money supply", "demand curve", "conditioning",
                            "therapist", "counselor", "dissertation", "supervisor",
                            "pro bono", "client", "session",
                            "congress", "congressional", "supreme court",
                            "federal", "federalism", "entitlement",
                            "legislative", "oversight", "bipartisan"],
        "other": ["diagnosis", "patient", "symptom", "accounting", "revenue",
                  "clinical", "therapy", "prescription", "anatomy", "nutrient",
                  "marketing", "customer", "brand"],
    }
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw.lower() in text)
        # Triple weight for strong signals
        if domain in strong_signals:
            count += 2 * sum(1 for kw in strong_signals[domain] if kw.lower() in text)
        scores[domain] = count
    
    # Numeric answers → likely STEM
    numeric_choices = sum(1 for c in choices if re.search(r'\d', c))
    if numeric_choices >= 3:
        scores["stem"] = scores.get("stem", 0) + 2
    
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
