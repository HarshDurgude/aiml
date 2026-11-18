import os
import csv
import random

os.makedirs("datasets", exist_ok=True)

# 1. heart.csv
with open(os.path.join("datasets", "heart.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"])
    random.seed(42)
    for _ in range(300):
        age = random.randint(29, 77)
        sex = random.choice([0, 1])
        cp = random.randint(0, 3)
        trestbps = random.randint(94, 200)
        chol = random.randint(126, 400)
        fbs = random.choice([0, 1])
        restecg = random.randint(0, 2)
        thalach = random.randint(71, 202)
        exang = random.choice([0, 1])
        oldpeak = round(random.uniform(0.0, 4.5), 1)
        slope = random.randint(0, 2)
        ca = random.randint(0, 3)
        thal = random.randint(1, 3)
        target = 1 if (age > 50 and chol > 240) or oldpeak > 2.0 else 0
        writer.writerow([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target])

# 2. Admission_Predict.csv
with open(os.path.join("datasets", "Admission_Predict.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Serial No.", "GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research", "target"])
    random.seed(42)
    for i in range(1, 401):
        gre = random.randint(290, 340)
        toefl = random.randint(92, 120)
        rating = random.randint(1, 5)
        sop = round(random.uniform(1.0, 5.0) * 2) / 2
        lor = round(random.uniform(1.0, 5.0) * 2) / 2
        cgpa = round(random.uniform(6.8, 9.9), 2)
        research = random.choice([0, 1])
        target = 1 if (gre >= 315 and cgpa >= 8.5) else 0
        writer.writerow([i, gre, toefl, rating, sop, lor, cgpa, research, target])

# 3. Housing.csv
with open(os.path.join("datasets", "Housing.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["price", "area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"])
    random.seed(42)
    for _ in range(350):
        area = random.randint(1650, 16200)
        bedrooms = random.randint(1, 6)
        bathrooms = random.randint(1, 4)
        stories = random.randint(1, 4)
        mainroad = random.choice(["yes", "no"])
        guestroom = random.choice(["yes", "no"])
        basement = random.choice(["yes", "no"])
        hotwater = random.choice(["yes", "no"])
        aircon = random.choice(["yes", "no"])
        parking = random.randint(0, 3)
        prefarea = random.choice(["yes", "no"])
        furnishing = random.choice(["furnished", "semi-furnished", "unfurnished"])
        price = int(area * 600 + bedrooms * 150000 + bathrooms * 300000 + (500000 if aircon == "yes" else 0) + random.randint(-200000, 200000))
        writer.writerow([price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwater, aircon, parking, prefarea, furnishing])

# 4. Mall_Customers.csv
with open(os.path.join("datasets", "Mall_Customers.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"])
    random.seed(42)
    for i in range(1, 201):
        gender = random.choice(["Male", "Female"])
        age = random.randint(18, 70)
        income = random.randint(15, 137)
        spending = random.randint(1, 99)
        writer.writerow([i, gender, age, income, spending])

# 5. wdbc.csv
with open(os.path.join("datasets", "wdbc.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
    writer.writerow(["id", "diagnosis"] + feature_names)
    random.seed(42)
    for i in range(1, 301):
        diag = random.choice(["M", "B"])
        base = 18.0 if diag == "M" else 12.0
        row = [842300 + i, diag]
        for _ in range(len(feature_names)):
            row.append(round(random.gauss(base, 2.5), 4))
        writer.writerow(row)

# 6. breast_cancer.csv
with open(os.path.join("datasets", "breast_cancer.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    features = [
        "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
        "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
        "radius error", "texture error", "perimeter error", "area error", "smoothness error",
        "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
        "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
        "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
    ]
    writer.writerow(features + ["target"])
    random.seed(42)
    for _ in range(300):
        target = random.choice([0, 1])
        base = 17.0 if target == 0 else 12.0
        row = [round(random.gauss(base, 2.0), 3) for _ in range(len(features))]
        row.append(target)
        writer.writerow(row)

print("All 6 datasets created successfully in datasets/")
