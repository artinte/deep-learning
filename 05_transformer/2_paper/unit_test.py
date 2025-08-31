import torch
from torchmetrics.text import BLEUScore


predictions_1 = ["the cat is on the mat"]
references_1 = [["the cat is on the mat"]]

bleu_calculator_1 = BLEUScore()
score_1 = bleu_calculator_1(predictions_1, references_1)
print(f"Prediction: {predictions_1}")
print(f"Reference: {references_1}")
print(f"BLEU Score: {score_1.item():.4f}")


