import argparse
import json
import sys
from .data import ROOT, TASKS, prepare


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="CyberLens: local English/Hinglish scam experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    train = commands.add_parser("train")
    train.add_argument("--backend", choices=["baseline", "word2vec", "subword", "transformer"], default="baseline")
    train.add_argument("--task", choices=["both", *TASKS], default="both")
    train.add_argument("--epochs", type=int)
    train.add_argument("--max-pairs", type=int, default=300000)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--accumulation", type=int, default=4)
    pred = commands.add_parser("predict")
    pred.add_argument("--text")
    pred.add_argument("--task", choices=["both", *TASKS], default="both")
    pred.add_argument("--backend", choices=["best", "baseline", "word2vec", "subword", "transformer"], default="best")
    pred.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        tasks = TASKS if args.task == "both" else (args.task,)
        if args.backend == "baseline":
            from .baseline import train
            train(tasks)
        elif args.backend == "transformer":
            from .transformer import train
            train(tasks, epochs=args.epochs or 1, batch_size=args.batch_size, accumulation=args.accumulation)
        else:
            from .embeddings import train
            train(tasks, mode=args.backend, epochs=args.epochs or 3, max_pairs=args.max_pairs)
    else:
        from .predict import Predictor
        predictor = Predictor(backend=args.backend)
        if args.interactive:
            print("Local inference. Enter :quit to exit. Do not paste private evidence into shared logs.")
            while True:
                try:
                    text = input("Message> ")
                    if text.strip() == ":quit":
                        break
                    print(json.dumps(predictor.predict(text, args.task), indent=2, ensure_ascii=False))
                except (EOFError, KeyboardInterrupt):
                    break
                except ValueError as exc:
                    print(exc)
        elif args.text:
            print(json.dumps(predictor.predict(args.text, args.task), indent=2, ensure_ascii=False))
        else:
            parser.error("Provide --text or --interactive")


if __name__ == "__main__":
    main()
