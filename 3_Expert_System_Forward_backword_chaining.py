# Practical 3: Rule-Based Expert System (Forward & Backward Chaining)

# Knowledge Base: (Premises, Conclusion)
rules = [
    (["Fever", "Cough"], "Flu"),
    (["Flu", "BodyPain"], "ViralInfection"),
    (["ViralInfection"], "Medicine"),
    (["Sneezing"], "Cold"),
    (["Cold"], "Rest")
]


# Forward Chaining (Data-Driven: Facts -> Conclusion)
def forward_chaining(facts):
    inferred = set(facts)
    changed = True
    while changed:
        changed = False
        for premises, conclusion in rules:
            if all(p in inferred for p in premises) and conclusion not in inferred:
                inferred.add(conclusion)
                print(f"Rule Fired: {premises} -> {conclusion}")
                changed = True
    return inferred


# Backward Chaining (Goal-Driven: Goal -> Hypothesis verification)
def backward_chaining(goal, facts, visited=None):
    if visited is None:
        visited = set()
    if goal in facts:
        return True
    if goal in visited:
        return False
    visited.add(goal)
    for premises, conclusion in rules:
        if conclusion == goal:
            if all(backward_chaining(p, facts, visited) for p in premises):
                print(f"Goal '{goal}' proved via rule premises {premises}")
                return True
    return False


if __name__ == "__main__":
    facts = {"Fever", "Cough", "BodyPain"}

    print("--- Forward Chaining ---")
    print("Inferred Facts:", forward_chaining(facts))

    print("\n--- Backward Chaining ---")
    goal = "Medicine"
    print(f"Is goal '{goal}' proved?:", backward_chaining(goal, facts))
