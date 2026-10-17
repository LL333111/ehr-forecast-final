"""Command-line entry points for the final pipeline."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Forecast first-recorded conditions from past EHR events."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser(
        "predict", help="Generate held-out probabilities without fitting."
    )
    predict.add_argument("--raw", type=Path, required=True, help="Supplied raw task directory.")
    predict.add_argument("--model", type=Path, default=Path("model/final_model.zip"))
    predict.add_argument("--cache", type=Path, default=Path("outputs/cache"))
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--split", choices=["val", "test"], default="test")
    prepare = commands.add_parser("prepare", help="Prepare numeric official-train inputs locally.")
    prepare.add_argument("--raw", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="Fit the fixed recipe on an authorized RunPod host.")
    train.add_argument(
        "--prepared", type=Path, required=True, help="Numeric training subdirectory only."
    )
    train.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export", help="Package fitted weights and local preprocessing.")
    export.add_argument(
        "--prepared", type=Path, required=True, help="Full local preparation directory."
    )
    export.add_argument("--trained", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Score locked official-validation predictions.")
    evaluate.add_argument("--raw", type=Path, required=True)
    evaluate.add_argument("--cache", type=Path, default=Path("outputs/cache/history"))
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "predict":
        from .bundle import unpack
        from .inference import predict_bundle
        from .io import sha

        model = unpack(args.model, args.cache / ("model-" + sha(args.model)[:16]))
        result = predict_bundle(args.raw, model, args.cache / "history", args.output, args.split)
    elif args.command == "prepare":
        from .training import prepare as prepare_inputs

        result = prepare_inputs(args.raw, args.output)
    elif args.command == "train":
        from .training import train as train_model

        result = train_model(args.prepared, args.output)
    elif args.command == "export":
        from .bundle import export as export_model

        result = export_model(args.prepared, args.trained, args.output)
    else:
        from .evaluation import evaluate as score_validation

        result = score_validation(args.raw, args.cache, args.predictions, args.output)
    print(json.dumps(result, indent=2))
