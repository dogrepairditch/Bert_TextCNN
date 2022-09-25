"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/9/23 15:07
@Email : handong_xu@163.com
"""
import torch
from utils.utils import *
from module.bert import TextCNN_Classifier
from train import test_data_loader,device
import torch.nn.functional as F

MODEL_PATH = os.path.join(MODEL_DIR,'best_model_state.ckpt')
model = TextCNN_Classifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
print('model load done')

def get_predictions(model, data_loader):
    model = model.eval()

    texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["texts"]

            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)


            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            print(texts)
            _, preds = torch.max(outputs, dim=1)
            print(preds)
            probs = F.softmax(outputs, dim=1)
            print(probs)

            texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return texts, predictions, prediction_probs, real_values

y_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)
print(y_test)