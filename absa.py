import gc
import os

import pandas as pd
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def get_aspect_sent(absa_tokenizer, absa_model, aspect, i, sentence, vid_id, prod, cat):
    sentiments = []
    inputs = absa_tokenizer(
        f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt"
    )
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    probs = probs.detach().numpy()[0]
    sentiments.append(
        {
            "id": i,
            "prod": prod,
            "cat": cat,
            "video_id": vid_id,
            "aspect": aspect,
            "negative": probs[0],
            "neutral": probs[1],
            "positive": probs[2],
        }
    )

    return sentiments


def process_instance(
    absa_tokenizer, absa_model, aspect, i, sentence, vid_id, prod, cat
):
    aspect_sent = get_aspect_sent(
        absa_tokenizer, absa_model, aspect, i, sentence, vid_id, prod, cat
    )
    aspect_sent_df = pd.DataFrame(aspect_sent)
    aspect_sent_df.to_csv(df_path, mode="a", header=not os.path.exists(df_path))
    del aspect_sent
    gc.collect()


absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification.from_pretrained(
    "yangheng/deberta-v3-base-absa-v1.1"
)

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = f"{dir_path}/transcript.csv"
aspect_sent = []
topics = [
    "features",
    "design",
    "fitness",
    "health",
    "comfort",
    "display",
    "comparison",
    "battery",
    "integration",
    "sleep",
    "sensor",
]


trans_df = pd.read_csv(data_path)
prod_df = trans_df[(trans_df["prod"] == "Watch")]

print(len(prod_df) * len(topics))

op_name = "aspect_sentiment_watch.csv"
df_path = f"{dir_path}/{op_name}"
start_write = False

if os.path.exists(df_path):
    df_current = pd.read_csv(df_path)
else:
    df_current = pd.DataFrame()

for i in range(len(prod_df)):
    sentence = prod_df["transcript"].iloc[i]
    vid_id = prod_df["video_id"].iloc[i]
    prod = prod_df["prod"].iloc[i]
    cat = prod_df["cat"].iloc[i]

    for aspect in topics:
        # check if file exsists
        if os.path.exists(df_path) and start_write == False and len(df_current) > 0:
            temp_df = df_current[
                (df_current["video_id"] == vid_id)
                & (df_current["id"] == i)
                & (df_current["aspect"] == aspect)
            ]
            if len(temp_df) > 0:
                continue
            else:
                start_write = True
                process_instance(
                    absa_tokenizer, absa_model, aspect, i, sentence, vid_id, prod, cat
                )

        else:
            process_instance(
                absa_tokenizer, absa_model, aspect, i, sentence, vid_id, prod, cat
            )


topics_ph = [
    "features",
    "build",
    "camera",
    "video",
    "updates",
    "connectivity",
    "charging",
    "power",
    "performance",
    "comparisons",
]
topics_earph = [
    "features",
    "wireless",
    "sound quality",
    "comfort",
    "technology",
    "design",
    "noise cancellation",
]
topics_watch = [
    "features",
    "design",
    "fitness",
    "health",
    "comfort",
    "display",
    "comparison",
    "battery",
    "integration",
    "sleep",
    "sensor",
]
topics_vr = [
    "performance",
    "experience",
    "gaming",
    "mixed reality",
    "Gyroscope",
    "audio",
    "visual",
    "technology",
    "specification",
]
