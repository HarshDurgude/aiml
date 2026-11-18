# Practical 3: Rule-Based Expert System (Forward Chaining & Backward Chaining)
# AIML Laboratory


class Rule:
    def __init__(self, premises, conclusion):
        self.premises = premises
        self.conclusion = conclusion


# ----------------------------
# Knowledge Base (Rules)
# ----------------------------
rules = [
    Rule(["Fever", "Cough"], "Flu"),
    Rule(["Flu", "BodyPain"], "ViralInfection"),
    Rule(["ViralInfection"], "Medicine"),
    Rule(["Sneezing"], "Cold"),
    Rule(["Cold"], "Rest")
]


# ==================================================
# FORWARD CHAINING (Data-Driven Inference)
# ==================================================
def forward_chaining(facts, rules):
    """
    Forward Chaining starts with known facts and applies rules to infer new facts until no more rules can be fired.
    """
    inferred = set(facts)
    print("\nInitial Facts:", inferred)

    changed = True
    while changed:
        changed = False
        for rule in rules:
            if all(p in inferred for p in rule.premises):
                if rule.conclusion not in inferred:
                    inferred.add(rule.conclusion)
                    print(f"Rule Fired: {rule.premises} -> {rule.conclusion}")
                    changed = True

    return inferred


# ==================================================
# BACKWARD CHAINING (Goal-Driven Inference)
# ==================================================
def backward_chaining(goal, facts, rules, visited=None):
    """
    Backward Chaining starts with a goal and works backward to see if known facts support it.
    """
    if visited is None:
        visited = set()

    if goal in facts:
        print(f"'{goal}' is a known fact.")
        return True

    if goal in visited:
        return False

    visited.add(goal)

    for rule in rules:
        if rule.conclusion == goal:
            print(f"\nTrying Rule: {rule.premises} -> {rule.conclusion}")
            success = True
            for premise in rule.premises:
                if not backward_chaining(premise, facts, rules, visited):
                    success = False
                    break
            if success:
                print(f"Goal '{goal}' proved successfully.")
                return True

    return False


# ==================================================
# Main Execution
# ==================================================
if __name__ == "__main__":
    facts = {"Fever", "Cough", "BodyPain"}

    print("=" * 50)
    print("===== FORWARD CHAINING (Data-Driven) =====")
    print("=" * 50)
    result = forward_chaining(facts, rules)

    print("\nFinal Inferred Facts:")
    for fact in sorted(result):
        print(f"  [+] {fact}")

    print("\n" + "=" * 50)
    print("===== BACKWARD CHAINING (Goal-Driven) =====")
    print("=" * 50)
    goal = "Medicine"
    print(f"Target Goal: {goal}")

    if backward_chaining(goal, facts, rules):
        print(f"\nResult: Goal '{goal}' is PROVED from current knowledge base.")
    else:
        print(f"\nResult: Goal '{goal}' CANNOT be proved.")
    print("=" * 50)
