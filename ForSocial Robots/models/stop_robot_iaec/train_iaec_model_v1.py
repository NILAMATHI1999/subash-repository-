#!/usr/bin/env python3

import copy
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn


ROOT = Path("/home/subash/thesis-social-robot")

MODEL_DIR = (
    ROOT
    / "models/stop_robot_iaec"
)

FEATURE_DIR = (
    MODEL_DIR
    / "features"
)

OUTPUT_PT = (
    MODEL_DIR
    / "stop_robot_iaec_v1.pt"
)

OUTPUT_ONNX = (
    MODEL_DIR
    / "stop_robot_iaec_v1.onnx"
)


STEPS = 2000
VALIDATE_EVERY = 100
LEARNING_RATE = 1e-3

SYNTHETIC_PER_CLASS = 24
REAL_PER_CLASS = 8

THRESHOLD = 0.5


torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)


def load_features(filename):
    return torch.from_numpy(
        np.load(
            FEATURE_DIR / filename
        ).astype(np.float32)
    )


synthetic_positive_microphone = load_features(
    "positive_train_microphone.npy"
)

synthetic_positive_reference = load_features(
    "positive_train_reference.npy"
)

synthetic_negative_microphone = load_features(
    "negative_train_microphone.npy"
)

synthetic_negative_reference = load_features(
    "negative_train_reference.npy"
)


test_positive_microphone = load_features(
    "positive_test_microphone.npy"
)

test_positive_reference = load_features(
    "positive_test_reference.npy"
)

test_negative_microphone = load_features(
    "negative_test_microphone.npy"
)

test_negative_reference = load_features(
    "negative_test_reference.npy"
)


real_positive_microphone = load_features(
    "real_positive_microphone.npy"
)

real_positive_reference = load_features(
    "real_positive_reference.npy"
)

real_negative_microphone = load_features(
    "real_negative_microphone.npy"
)

real_negative_reference = load_features(
    "real_negative_reference.npy"
)


print(
    "Synthetic positive train:",
    tuple(
        synthetic_positive_microphone.shape
    ),
)

print(
    "Synthetic negative train:",
    tuple(
        synthetic_negative_microphone.shape
    ),
)

print(
    "Synthetic positive test:",
    tuple(
        test_positive_microphone.shape
    ),
)

print(
    "Synthetic negative test:",
    tuple(
        test_negative_microphone.shape
    ),
)

print(
    "Real positive train:",
    tuple(
        real_positive_microphone.shape
    ),
)

print(
    "Real negative train:",
    tuple(
        real_negative_microphone.shape
    ),
)


class EchoAwareKeywordModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.microphone_projection = nn.Linear(
            96,
            48,
        )

        self.reference_projection = nn.Linear(
            96,
            48,
        )

        self.microphone_norm = nn.LayerNorm(
            48
        )

        self.reference_norm = nn.LayerNorm(
            48
        )

        self.activation = nn.ReLU()

        # Four 48-dimensional streams over 16 frames:
        # microphone, reference, difference and interaction.
        self.fusion = nn.Linear(
            16 * 48 * 4,
            128,
        )

        self.fusion_norm = nn.LayerNorm(
            128
        )

        self.dropout = nn.Dropout(
            0.20
        )

        self.output = nn.Linear(
            128,
            1,
        )

    def forward(
        self,
        microphone_features,
        reference_features,
    ):
        microphone_hidden = self.activation(
            self.microphone_norm(
                self.microphone_projection(
                    microphone_features
                )
            )
        )

        reference_hidden = self.activation(
            self.reference_norm(
                self.reference_projection(
                    reference_features
                )
            )
        )

        difference = (
            microphone_hidden
            - reference_hidden
        )

        interaction = (
            microphone_hidden
            * reference_hidden
        )

        combined = torch.cat(
            (
                microphone_hidden,
                reference_hidden,
                difference,
                interaction,
            ),
            dim=2,
        )

        combined = combined.flatten(
            start_dim=1
        )

        hidden = self.activation(
            self.fusion_norm(
                self.fusion(
                    combined
                )
            )
        )

        hidden = self.dropout(
            hidden
        )

        return self.output(
            hidden
        )


class ExportModel(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(
        self,
        microphone_features,
        reference_features,
    ):
        return torch.sigmoid(
            self.classifier(
                microphone_features,
                reference_features,
            )
        )


model = EchoAwareKeywordModel()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

loss_function = (
    nn.BCEWithLogitsLoss()
)


@torch.no_grad()
def predict_scores(
    microphone,
    reference,
):
    model.eval()

    scores = []

    for start in range(
        0,
        len(microphone),
        256,
    ):
        end = start + 256

        logits = model(
            microphone[start:end],
            reference[start:end],
        ).squeeze(1)

        scores.append(
            torch.sigmoid(logits)
        )

    return torch.cat(scores)


@torch.no_grad()
def evaluate(label):
    positive_scores = predict_scores(
        test_positive_microphone,
        test_positive_reference,
    )

    negative_scores = predict_scores(
        test_negative_microphone,
        test_negative_reference,
    )

    recall = float(
        (
            positive_scores
            >= THRESHOLD
        ).float().mean()
    )

    false_positive_rate = float(
        (
            negative_scores
            >= THRESHOLD
        ).float().mean()
    )

    accuracy = float(
        (
            (
                positive_scores
                >= THRESHOLD
            ).float().sum()
            +
            (
                negative_scores
                < THRESHOLD
            ).float().sum()
        )
        /
        (
            len(positive_scores)
            + len(negative_scores)
        )
    )

    print(
        f"{label}: "
        f"recall={recall:.3f}, "
        f"fpr={false_positive_rate:.3f}, "
        f"accuracy={accuracy:.3f}, "
        f"positive_mean="
        f"{float(positive_scores.mean()):.4f}, "
        f"negative_max="
        f"{float(negative_scores.max()):.4f}"
    )

    model.train()

    return {
        "recall": recall,
        "fpr": false_positive_rate,
        "accuracy": accuracy,
    }


@torch.no_grad()
def evaluate_real_training_sanity():
    positive_scores = predict_scores(
        real_positive_microphone,
        real_positive_reference,
    )

    negative_scores = predict_scores(
        real_negative_microphone,
        real_negative_reference,
    )

    real_recall = float(
        (
            positive_scores
            >= THRESHOLD
        ).float().mean()
    )

    real_fpr = float(
        (
            negative_scores
            >= THRESHOLD
        ).float().mean()
    )

    print(
        "Real training sanity: "
        f"recall={real_recall:.3f}, "
        f"fpr={real_fpr:.3f}, "
        f"positive_mean="
        f"{float(positive_scores.mean()):.4f}, "
        f"negative_max="
        f"{float(negative_scores.max()):.4f}"
    )

    model.train()


best_state = copy.deepcopy(
    model.state_dict()
)

best_selection_score = float("inf")


for step in range(
    1,
    STEPS + 1,
):
    synthetic_positive_indices = (
        torch.randint(
            len(
                synthetic_positive_microphone
            ),
            (
                SYNTHETIC_PER_CLASS,
            ),
        )
    )

    synthetic_negative_indices = (
        torch.randint(
            len(
                synthetic_negative_microphone
            ),
            (
                SYNTHETIC_PER_CLASS,
            ),
        )
    )

    real_positive_indices = (
        torch.randint(
            len(
                real_positive_microphone
            ),
            (
                REAL_PER_CLASS,
            ),
        )
    )

    real_negative_indices = (
        torch.randint(
            len(
                real_negative_microphone
            ),
            (
                REAL_PER_CLASS,
            ),
        )
    )

    microphone_batch = torch.cat(
        (
            synthetic_positive_microphone[
                synthetic_positive_indices
            ],

            real_positive_microphone[
                real_positive_indices
            ],

            synthetic_negative_microphone[
                synthetic_negative_indices
            ],

            real_negative_microphone[
                real_negative_indices
            ],
        ),
        dim=0,
    )

    reference_batch = torch.cat(
        (
            synthetic_positive_reference[
                synthetic_positive_indices
            ],

            real_positive_reference[
                real_positive_indices
            ],

            synthetic_negative_reference[
                synthetic_negative_indices
            ],

            real_negative_reference[
                real_negative_indices
            ],
        ),
        dim=0,
    )

    positive_count = (
        SYNTHETIC_PER_CLASS
        + REAL_PER_CLASS
    )

    negative_count = (
        SYNTHETIC_PER_CLASS
        + REAL_PER_CLASS
    )

    labels = torch.cat(
        (
            torch.ones(
                positive_count
            ),

            torch.zeros(
                negative_count
            ),
        )
    )

    model.train()

    logits = model(
        microphone_batch,
        reference_batch,
    ).squeeze(1)

    loss = loss_function(
        logits,
        labels,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % VALIDATE_EVERY == 0:
        metrics = evaluate(
            f"Step {step}"
        )

        # Missed commands receive twice the penalty
        # of false activations during checkpoint choice.
        selection_score = (
            2.0
            * (
                1.0
                - metrics["recall"]
            )
            + metrics["fpr"]
        )

        if (
            selection_score
            < best_selection_score
        ):
            best_selection_score = (
                selection_score
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                best_state,
                OUTPUT_PT,
            )

            print(
                "  Saved best iAEC checkpoint"
            )


model.load_state_dict(
    best_state
)

final_metrics = evaluate(
    "Selected iAEC V1"
)

evaluate_real_training_sanity()


export_model = ExportModel(
    model
).eval()

dummy_microphone = torch.randn(
    1,
    16,
    96,
)

dummy_reference = torch.randn(
    1,
    16,
    96,
)


torch.onnx.export(
    export_model,
    (
        dummy_microphone,
        dummy_reference,
    ),
    str(OUTPUT_ONNX),
    input_names=[
        "microphone_features",
        "reference_features",
    ],
    output_names=[
        "score",
    ],
    dynamic_axes={
        "microphone_features": {
            0: "batch",
        },
        "reference_features": {
            0: "batch",
        },
        "score": {
            0: "batch",
        },
    },
    opset_version=14,
    dynamo=False,
)


onnx.checker.check_model(
    onnx.load(
        str(OUTPUT_ONNX)
    )
)


print(
    "Saved:",
    OUTPUT_PT,
)

print(
    "Saved:",
    OUTPUT_ONNX,
)

print(
    "Dual-input iAEC training complete."
)
