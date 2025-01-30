import json
import statistics

checks = []
scores = []
total = 0
true = 0

with open("bdd100k_labels_images_val.json", "r") as f:
    ground_truth = json.load(f)

with open("preds.json", "r") as f:
    preds = json.load(f)

pred_class = preds["pred_class"]
pred_score = preds["pred_score"]
true_weather = [instance["attributes"]["weather"] for instance in ground_truth]

for i in range(len(true_weather)):
    checks.append((true_weather[i], pred_class[i]))
    scores.append((true_weather[i], pred_score[i]))
    if true_weather[i] == pred_class[i]:
        total += 1
        true += 1
    else:
        total += 1


print("Total: ", total)
print("True: ", true)
print("Accuracy: ", true/total)

weather_classes = set([true for true, pred, in checks])
metrics = {cls: {"True": 0, "False": 0, "Scores": []} for cls in weather_classes}

for true, pred in checks:
    for cls in weather_classes:
        if true == pred and cls == true:
            metrics[cls]["True"] += 1
        elif cls == true:
            metrics[cls]["False"] += 1

for weather, score in scores:
    for cls in weather_classes:
        if weather == cls:
            metrics[cls]["Scores"].append(score)


for cls, counts, in metrics.items():
    print("{}: {}".format(cls, counts))
    counts["Accuracy"] = (counts["True"]/(counts["True"] + counts["False"]))
    counts["Score_avg"] = statistics.mean(counts["Scores"])


for cls, counts in metrics.items():
    print(f"Weather Class: {cls}")
    print(f"  Correct Predictions: {counts['True']}")
    print(f"  False Predictions: {counts['False']}")
    print(f"  Accuracy: {counts['Accuracy']*100:.2f}%")
    print(f"  Average Prediction Certainty: {counts['Score_avg']*100:.2f}%")