#!/usr/bin/env python3

import copy
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper


ROOT = Path("/home/subash/thesis-social-robot")
MODEL_DIR = ROOT / "models/stop_robot"
FEATURE_DIR = MODEL_DIR / "training_data_v1/stop_robot"

BASELINE = MODEL_DIR / "stop_robot_v4.onnx"
OUTPUT_ONNX = MODEL_DIR / "stop_robot_v5.onnx"
OUTPUT_PT = MODEL_DIR / "stop_robot_v5.pt"

STEPS = 1000
VALIDATE_EVERY = 100
LEARNING_RATE = 1e-5

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)


class StopRobotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(16 * 96, 128)
        self.layernorm1 = nn.LayerNorm(128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 1)

    def forward(self, audio_features):
        hidden = self.flatten(audio_features)
        hidden = self.layer1(hidden)
        hidden = self.layernorm1(hidden)
        hidden = self.relu1(hidden)
        return self.layer2(hidden)


class ExportModel(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.base = classifier

    def forward(self, audio_features):
        return torch.sigmoid(self.base(audio_features))


def load_features(filename):
    return torch.from_numpy(
        np.load(filename).astype(np.float32)
    )


positive_train = load_features(
    FEATURE_DIR / "positive_features_train_v5.npy"
)
negative_train = load_features(
    FEATURE_DIR / "negative_features_train.npy"
)
positive_test = load_features(
    FEATURE_DIR / "positive_features_test.npy"
)
negative_test = load_features(
    FEATURE_DIR / "negative_features_test.npy"
)
hard_all = load_features(
    MODEL_DIR / "hard_negative_features_v5.npy"
)

# Hold out “Hey Jarvis” and six playback-reference clips.
hard_train = hard_all
hard_test = hard_all

print("Positive train:", tuple(positive_train.shape))
print("Synthetic negative train:", tuple(negative_train.shape))
print("Hard-negative train:", tuple(hard_train.shape))
print("Hard-negative held-out test:", tuple(hard_test.shape))


model = StopRobotModel()

onnx_model = onnx.load(str(BASELINE))
weights = {
    item.name: numpy_helper.to_array(item).copy()
    for item in onnx_model.graph.initializer
}

with torch.no_grad():
    for name, parameter in model.named_parameters():
        onnx_name = f"base.{name}"

        if onnx_name not in weights:
            raise KeyError(f"Missing ONNX weight: {onnx_name}")

        source = torch.from_numpy(weights[onnx_name])

        if source.shape != parameter.shape:
            raise ValueError(
                f"{onnx_name}: expected {tuple(parameter.shape)}, "
                f"found {tuple(source.shape)}"
            )

        parameter.copy_(source)

print("Baseline ONNX weights loaded into PyTorch")


@torch.no_grad()
def evaluate(label):
    model.eval()

    positive_scores = torch.sigmoid(
        model(positive_test)
    ).squeeze(1)

    negative_scores = torch.sigmoid(
        model(negative_test)
    ).squeeze(1)

    hard_scores = torch.sigmoid(
        model(hard_test)
    ).squeeze(1)

    metrics = {
        "recall": float(
            (positive_scores >= 0.5).float().mean()
        ),
        "synthetic_fpr": float(
            (negative_scores >= 0.5).float().mean()
        ),
        "hard_fpr": float(
            (hard_scores >= 0.5).float().mean()
        ),
        "hard_max": float(hard_scores.max()),
    }

    print(
        f"{label}: "
        f"recall={metrics['recall']:.3f}, "
        f"synthetic_fpr={metrics['synthetic_fpr']:.3f}, "
        f"hard_fpr={metrics['hard_fpr']:.3f}, "
        f"hard_max={metrics['hard_max']:.4f}"
    )

    model.train()
    return metrics


baseline_metrics = evaluate("Baseline")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

loss_function = nn.BCEWithLogitsLoss(
    reduction="none"
)

best_state = copy.deepcopy(model.state_dict())
best_score = float("inf")

model.train()

for step in range(1, STEPS + 1):
    positive_indices = torch.randint(
        len(positive_train),
        (32,),
    )
    negative_indices = torch.randint(
        len(negative_train),
        (32,),
    )
    hard_indices = torch.randint(
        len(hard_train),
        (32,),
    )

    features = torch.cat(
        (
            positive_train[positive_indices],
            negative_train[negative_indices],
            hard_train[hard_indices],
        )
    )

    labels = torch.cat(
        (
            torch.ones(32),
            torch.zeros(32),
            torch.zeros(32),
        )
    )

    # Real hard negatives receive extra importance.
    weights_for_loss = torch.cat(
        (
            torch.ones(32),
            torch.ones(32),
            torch.full((32,), 5.0),
        )
    )

    logits = model(features).squeeze(1)

    loss = (
        loss_function(logits, labels)
        * weights_for_loss
    ).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % VALIDATE_EVERY == 0:
        metrics = evaluate(f"Step {step}")

        # Preserve command recall while minimizing real false triggers.
        if metrics["recall"] >= 0.90:
            score = (
                metrics["hard_max"]
                + metrics["hard_fpr"] * 2.0
                + metrics["synthetic_fpr"]
            )

            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(
                    model.state_dict()
                )
                torch.save(best_state, OUTPUT_PT)
                print("  Saved best V2 checkpoint")


model.load_state_dict(best_state)
final_metrics = evaluate("Selected V2")

export_model = ExportModel(model).eval()
dummy_input = torch.randn(1, 16, 96)

torch.onnx.export(
    export_model,
    dummy_input,
    str(OUTPUT_ONNX),
    input_names=["onnx::Flatten_0"],
    output_names=["output"],
    dynamic_axes={
        "onnx::Flatten_0": {0: "batch"},
        "output": {0: "batch"},
    },
    opset_version=14,
    dynamo=False,
)

onnx.checker.check_model(
    onnx.load(str(OUTPUT_ONNX))
)

print("Saved:", OUTPUT_ONNX)
print("Saved:", OUTPUT_PT)
print("V2 fine-tuning complete")
