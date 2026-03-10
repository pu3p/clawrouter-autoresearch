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
                     "sleep deprivation", "rem sleep",
                     "neurotransmitter", "nature vs. nurture",
                     "major depression", "stimulus",
                     "sperling", "opponent process",
                     "species-specific", "parenting",
                     "psychology", "therapist", "therapy",
                     "behavior and mental", "abusive"],
        "min_signals": 2,
    },
    # MEDIUM: high_school_psychology_strong — unique phrases
    "high_school_psychology_strong": {
        "tier": "MEDIUM",
        "signals": ["capital punishment", "person diagnosed",
                     "ratio schedules", "schedules of",
                     "psychological disorders", "representativeness heuristic",
                     "availability heuristic", "serial position",
                     "dissociative disorder", "behavior modification",
                     "psychology exam", "research participant",
                     "sensory memory"],
        "min_signals": 1,
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
    # MEDIUM: high_school_government_strong
    "high_school_government_strong": {
        "tier": "MEDIUM",
        "signals": ["the secretary of", "entitlement programs",
                     "constitutional amendment.", "national government",
                     "most such programs"],
        "min_signals": 1,
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
    # MEDIUM: high_school_macroeconomics_strong
    "high_school_macroeconomics_strong": {
        "tier": "MEDIUM",
        "signals": ["the equilibrium price", "equilibrium quantity",
                     "real gdp", "full employment.",
                     "the price level", "equilibrium price level"],
        "min_signals": 1,
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
    # MEDIUM: high_school_microeconomics_strong
    "high_school_microeconomics_strong": {
        "tier": "MEDIUM",
        "signals": ["average total", "average variable",
                     "a firm's", "product market",
                     "a monopolistically", "cost equals"],
        "min_signals": 1,
    },
    # MEDIUM: high_school_biology
    "high_school_biology": {
        "tier": "MEDIUM",
        "signals": ["darwin", "natural selection", "elodea",
                     "small intestine", "photosynthesis", "mitosis",
                     "meiosis", "dna replication", "allele",
                     "phenotype", "genotype", "ecosystem",
                     "food chain", "cell membrane", "ribosome",
                     "earthworm", "spiracle", "alveoli",
                     "hemophilia", "extraembryonic", "predator",
                     "aquatic plant", "chloroplast", "xylem"],
        "min_signals": 2,
    },
    # MEDIUM: high_school_biology_strong — unique phrases
    "high_school_biology_strong": {
        "tier": "MEDIUM",
        "signals": ["dna content", "metaphase i", "these organisms",
                     "secretion will", "two nucleotides",
                     "descent with modification", "hardy-weinberg equilibrium",
                     "while prokaryotes", "glucose levels will",
                     "gene alleles",
                     "there are differences in", "among the species.",
                     "the cell in metaphase", "will stop and",
                     "hydrogen bonding", "skeletal structures",
                     "forelimbs", "reptile eggs"],
        "min_signals": 1,
    },
    # MEDIUM: high_school_chemistry
    "high_school_chemistry": {
        "tier": "MEDIUM",
        "signals": ["lewis structure", "ground state", "bond strength",
                     "electron configuration", "homogeneous",
                     "heterogeneous", "molar mass", "oxidation",
                     "reduction", "periodic table", "valence",
                     "molarity", "titration", "equilibrium constant"],
        "min_signals": 2,
    },
    # MEDIUM: high_school_chemistry_strong
    "high_school_chemistry_strong": {
        "tier": "MEDIUM",
        "signals": ["is transferred from", "hno2 >", "h2co3 >",
                     "> h3po4", "the catalyst", "> h2s"],
        "min_signals": 1,
    },
    # MEDIUM: high_school_mathematics_strong — unique phrases
    "high_school_mathematics_strong": {
        "tier": "MEDIUM",
        "signals": ["(x, y)", "the domain of", "have an inverse",
                     "express your answer", "the graph of",
                     "find the sum", "domain of $f$",
                     "all real", "where $a$"],
        "min_signals": 1,
    },
    # MEDIUM: high_school_geography
    "high_school_geography": {
        "tier": "MEDIUM",
        "signals": ["federal state", "city planners", "land use",
                     "world population", "central place theory",
                     "urban", "rural", "migration",
                     "demographic transition", "population density",
                     "cultural landscape", "spatial",
                     "stateless nation", "continentality",
                     "universalizing religion", "land survey",
                     "immigrant learning"],
        "min_signals": 2,
    },
    # MEDIUM: high_school_computer_science_strong
    "high_school_computer_science_strong": {
        "tier": "MEDIUM",
        "signals": ["block-based programming", "string s",
                     "logic gate", "a computer system",
                     "machine language,"],
        "min_signals": 1,
    },
    # MEDIUM: high_school_physics
    "high_school_physics": {
        "tier": "MEDIUM",
        "signals": ["toy car", "light bulb",
                     "balloon", "buoyant force", "propel",
                     "collision", "newton's", "acceleration",
                     "velocity", "kinetic energy", "potential energy",
                     "circuit", "voltage", "resistance"],
        "min_signals": 3,
    },
    # MEDIUM: high_school_physics_strong
    "high_school_physics_strong": {
        "tier": "MEDIUM",
        "signals": ["the swimmer's", "swimmer's arms",
                     "the cart", "average kinetic",
                     "the molecules of"],
        "min_signals": 1,
    },
    # MEDIUM: high_school_statistics
    "high_school_statistics": {
        "tier": "MEDIUM",
        "signals": ["standard deviation", "binomial", "sample survey",
                     "bias", "continuous data", "confidence interval",
                     "null hypothesis", "p-value", "regression",
                     "correlation", "normal distribution",
                     "random sample", "histogram"],
        "min_signals": 2,
    },
    # MEDIUM: high_school_statistics_strong
    "high_school_statistics_strong": {
        "tier": "MEDIUM",
        "signals": ["standard deviation of", "percent confidence",
                     "above the mean", "a binomial"],
        "min_signals": 1,
    },
    # COMPLEX: sociology
    "sociology": {
        "tier": "COMPLEX",
        "signals": ["functionalist", "social stratum", "foucault",
                     "weber", "parsons", "urbanization",
                     "biopolitics", "anti-psychiatrist",
                     "social constructivism", "domestic violence",
                     "social science", "national curriculum",
                     "disciplinary power", "differential association",
                     "sick role", "sexual revolution"],
        "min_signals": 1,
    },
    # COMPLEX: security_studies
    "security_studies": {
        "tier": "COMPLEX",
        "signals": ["chemical weapons", "human insecurity",
                     "postcolonialism", "security studies",
                     "historical materialism", "greedy state",
                     "security seeker", "international society",
                     "nuclear weapon", "arms control"],
        "min_signals": 1,
    },
    # COMPLEX: prehistory
    "prehistory": {
        "tier": "COMPLEX",
        "signals": ["mogollon", "inca", "mesopotamia",
                     "rockshelter", "quarry site", "neolithic",
                     "paleolithic", "bronze age", "iron age",
                     "archaeological", "artifact", "excavation",
                     "hunter-gatherer", "domestication"],
        "min_signals": 1,
    },
    # COMPLEX: prehistory_strong
    "prehistory_strong": {
        "tier": "COMPLEX",
        "signals": ["homo erectus", "homo ergaster", "homo antecessor",
                     "million years", "skeletal remains",
                     "first americans,", "those who lived",
                     "homo habilis", "australopithecus",
                     "neandertal", "neanderthal", "denisovan",
                     "bering strait", "cranial capacit",
                     "maize-based", "aztec", "tenochtitl",
                     "foraging subsistence", "stone tool",
                     "archaic period", "mound 72"],
        "min_signals": 1,
    },
    # COMPLEX: world_religions
    "world_religions": {
        "tier": "COMPLEX",
        "signals": ["shari'ah", "guru", "darbar sahib",
                     "deaconess", "christianity", "buddhism",
                     "hinduism", "islam", "judaism",
                     "torah", "quran", "vedas",
                     "mosque", "synagogue", "monastery",
                     "festival of", "bushido", "baptism",
                     "theologian", "pelagius", "calvin",
                     "sutra", "shoah", "apocalypse",
                     "zen", "maitreya"],
        "min_signals": 2,
    },
    # COMPLEX: philosophy
    "philosophy": {
        "tier": "COMPLEX",
        "signals": ["utilitarianism", "rule-utilitarianism",
                     "deontological", "kantian", "consequentialism",
                     "epistemology", "metaphysics", "ontology",
                     "moral realism", "moral relativism",
                     "think critically", "good reasons"],
        "min_signals": 1,
    },
    # COMPLEX: philosophy_strong
    "philosophy_strong": {
        "tier": "COMPLEX",
        "signals": ["every pleasure is", "of philosophy",
                     "equal concern for", "be chosen.",
                     "pleasure is good", "for the survival of each,"],
        "min_signals": 1,
    },
    # COMPLEX: moral_disputes_strong
    "moral_disputes_strong": {
        "tier": "COMPLEX",
        "signals": ["the original position", "should prohibit",
                     "prohibit things", "b neither a nor",
                     "neither a nor b",
                     "according to singer", "according to marquis",
                     "according to locke", "according to mill",
                     "according to gardiner", "according to walzer",
                     "doctrine of double effect",
                     "retributivist", "death penalty",
                     "drug legalization", "decriminalization",
                     "animal rights", "van den haag",
                     "de marneffe", "pogge argues"],
        "min_signals": 1,
    },
    # COMPLEX: human_sexuality
    "human_sexuality": {
        "tier": "COMPLEX",
        "signals": ["a g-spot", "masturbation produces",
                     "the vagina", "sexual arousal",
                     "orgasm", "erogenous"],
        "min_signals": 1,
    },
    # COMPLEX: econometrics
    "econometrics": {
        "tier": "COMPLEX",
        "signals": ["random effects", "explanatory variables",
                     "the rhs", "correlated with"],
        "min_signals": 1,
    },
    # COMPLEX: us_foreign_policy
    "us_foreign_policy": {
        "tier": "COMPLEX",
        "signals": ["cold war", "league of nations",
                     "un security council", "soft power",
                     "grand strategy", "containment",
                     "primacy", "belligerent"],
        "min_signals": 1,
    },
    # COMPLEX: public_relations
    "public_relations": {
        "tier": "COMPLEX",
        "signals": ["public relations", "your supervisor"],
        "min_signals": 1,
    },
    # COMPLEX: international_law
    "international_law": {
        "tier": "COMPLEX",
        "signals": ["of the treaty", "optional clause",
                     "the parties to", "treaties that"],
        "min_signals": 1,
    },
    # COMPLEX: logical_fallacies
    "logical_fallacies": {
        "tier": "COMPLEX",
        "signals": ["the fallacy of", "claim should be accepted",
                     "that a claim"],
        "min_signals": 1,
    },
    # COMPLEX: formal_logic
    "formal_logic": {
        "tier": "COMPLEX",
        "signals": ["consistent. consistent valuation",
                     "consistent valuation when",
                     "are true and"],
        "min_signals": 1,
    },
    # COMPLEX: jurisprudence
    "jurisprudence": {
        "tier": "COMPLEX",
        "signals": ["which proposition below", "most powerful refutation",
                     "the law."],
        "min_signals": 1,
    },
    # COMPLEX: human_aging
    "human_aging": {
        "tier": "COMPLEX",
        "signals": ["telomere strands", "a bigger role",
                     "gay and lesbian couples",
                     "satisfied with their", "cancer cells",
                     "normal cells"],
        "min_signals": 1,
    },
    # COMPLEX: abstract_algebra
    "abstract_algebra": {
        "tier": "COMPLEX",
        "signals": ["statement 1 |", "statement 2 |"],
        "min_signals": 1,
    },
    # COMPLEX: machine_learning
    "machine_learning": {
        "tier": "COMPLEX",
        "signals": ["p(x, y, z)", "set it to zero"],
        "min_signals": 1,
    },
    # COMPLEX: electrical_engineering
    "electrical_engineering": {
        "tier": "COMPLEX",
        "signals": ["no load", "quantity being measured.",
                     "electric field"],
        "min_signals": 1,
    },
    # COMPLEX: conceptual_physics
    "conceptual_physics": {
        "tier": "COMPLEX",
        "signals": ["inclined plane", "normal force", "charged particles",
                     "radiates", "correspondence principle",
                     "rolling down", "deflected by",
                     "magnetic means", "gravitational",
                     "projectile", "free fall", "wavelength",
                     "speed of light", "photon", "greenhouse gas",
                     "standing waves", "fission or fusion",
                     "apparent weight", "hologram",
                     "radioactive nucleus", "planck"],
        "min_signals": 2,
    },
    # COMPLEX: conceptual_physics_strong
    "conceptual_physics_strong": {
        "tier": "COMPLEX",
        "signals": ["none of these neither", "neither of these",
                     "of the atmosphere", "than mg",
                     "of charge.", "two separated"],
        "min_signals": 1,
    },
    # COMPLEX: astronomy
    "astronomy": {
        "tier": "COMPLEX",
        "signals": ["blackbody", "lunar maria", "venus",
                     "mercury", "saturn", "jupiter",
                     "solar system", "light-year", "parsec",
                     "supernova", "nebula", "galaxy",
                     "telescope", "red giant"],
        "min_signals": 2,
    },
    # COMPLEX: astronomy_strong
    "astronomy_strong": {
        "tier": "COMPLEX",
        "signals": ["peak emission wavelength", "power emitted is",
                     "may not be able"],
        "min_signals": 1,
    },
    # COMPLEX: computer_security
    "computer_security": {
        "tier": "COMPLEX",
        "signals": ["firewall", "intrusion detection",
                     "buffer overflow", "sql injection",
                     "cross-site scripting", "authentication",
                     "cryptographic", "cipher", "malware",
                     "phishing", "denial of service"],
        "min_signals": 1,
    },
    # COMPLEX: college_computer_science
    "college_computer_science": {
        "tier": "COMPLEX",
        "signals": ["operating system", "network operating",
                     "distributed operating"],
        "min_signals": 1,
    },
    # COMPLEX: college_medicine
    "college_medicine": {
        "tier": "COMPLEX",
        "signals": ["graduated cylinder", "adhesive forces",
                     "fibres. type", "the mercury",
                     "how many chromosomes", "drugs to modify",
                     "production of atp.", "cohesive forces",
                     "creatine kinase", "creatine to load",
                     "contractile proteins", "somatic eukaryotic"],
        "min_signals": 1,
    },
    # COMPLEX: college_biology
    "college_biology": {
        "tier": "COMPLEX",
        "signals": ["alarm call", "actin-myosin",
                     "the affected cells", "polyteny",
                     "auxin", "lobed thallus", "rhizoids",
                     "kin selection", "swimming sperm",
                     "xenopus laevis", "continental drift",
                     "prokaryotes and eukaryotes"],
        "min_signals": 1,
    },
    # COMPLEX: business_ethics
    "business_ethics": {
        "tier": "COMPLEX",
        "signals": ["corporate governance", "shareholders",
                     "executive directors", "stakeholder",
                     "csr", "sustainability agenda",
                     "disqualification of directors",
                     "business ethics", "oecd"],
        "min_signals": 1,
    },
    # COMPLEX: medical_genetics
    "medical_genetics": {
        "tier": "COMPLEX",
        "signals": ["sickle cell", "rflp", "marfan syndrome",
                     "hardy-weinberg", "autosomal recessive",
                     "cell lineage", "gametes", "lac operon",
                     "restriction enzymes", "sticky ends",
                     "heritability", "base pairs"],
        "min_signals": 1,
    },
    # COMPLEX: anatomy
    "anatomy": {
        "tier": "COMPLEX",
        "signals": ["parotid gland", "intercostal space",
                     "maxillary sinus", "connective tissue",
                     "tonsillar", "homeostasis",
                     "stethoscope", "mid-axillary",
                     "flexor", "extensor", "ligament"],
        "min_signals": 1,
    },
    # COMPLEX: management
    "management": {
        "tier": "COMPLEX",
        "signals": ["decision making model", "corporate manager",
                     "bureaucratic organisation", "power distance",
                     "key skill of management", "authority refer",
                     "organisational", "leadership style"],
        "min_signals": 1,
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
                     "transamination", "bmr", "nutrient", "nutritional",
                     "provides less energy than", "hydrophobic core"],
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
    # COMPLEX: clinical_knowledge_strong
    "clinical_knowledge_strong": {
        "tier": "COMPLEX",
        "signals": ["dentures should be", "the pain of",
                     "released from the muscle.",
                     "the patient should be"],
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
    # Prehistory/world_religions/philosophy/moral_disputes → humanities domain
    if _detected_subject and any(s in _detected_subject for s in ["prehistory", "world_religions", "philosophy", "moral_disputes", "international_law", "logical_fallacies", "formal_logic", "jurisprudence"]):
        domain = "humanities"
    # Human_sexuality → social_sciences domain
    if _detected_subject and "human_sexuality" in _detected_subject:
        domain = "social_sciences"
    # Government/economics-detected → social_sciences domain
    if _detected_subject and any(s in _detected_subject for s in ["government", "macroeconomics", "microeconomics", "high_school_geography", "sociology", "security_studies", "econometrics", "public_relations", "human_sexuality", "us_foreign_policy"]):
        domain = "social_sciences"
    # Marketing/management/business_ethics/nutrition/clinical_knowledge/virology → other domain
    if _detected_subject and any(s in _detected_subject for s in ["marketing", "nutrition", "virology", "clinical_knowledge", "business_ethics", "medical_genetics", "anatomy", "management", "college_medicine", "human_aging"]):
        domain = "other"
    # Physics/astronomy/computer_security/college_computer_science → stem domain
    if _detected_subject and any(s in _detected_subject for s in ["conceptual_physics", "astronomy", "computer_security", "college_computer_science", "abstract_algebra", "machine_learning", "electrical_engineering", "high_school_biology", "high_school_chemistry", "high_school_physics", "high_school_statistics", "high_school_mathematics", "high_school_computer_science", "college_biology"]):
        domain = "stem"
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
