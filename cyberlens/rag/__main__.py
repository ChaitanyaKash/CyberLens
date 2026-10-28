"""python -m cyberlens.rag {build,ask,eval,export-mobile}"""
import argparse
import json
import sys
from .config import CORPUS, EMBED_MODEL, INDEX, LLM_MODEL, ROOT


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m cyberlens.rag", description="Cited guidance for cyber-fraud victims")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="parse, chunk and embed the corpus (GPU optional; see modal_app for remote)")
    build.add_argument("--embedder", default=EMBED_MODEL, help=f"'{EMBED_MODEL}' or 'hash' (no model, for testing)")
    build.add_argument("--out", default=str(INDEX))
    ask = commands.add_parser("ask")
    ask.add_argument("text")
    ask.add_argument("--subtype", help="complaint label; if omitted, the CyberLens classifier predicts it")
    ask.add_argument("--date", help="incident date YYYY-MM-DD (default today)")
    ask.add_argument("--llm", choices=["ollama", "none"], default="ollama")
    ask.add_argument("--model", default=LLM_MODEL)
    ask.add_argument("--index", default=str(INDEX))
    ev = commands.add_parser("eval")
    ev.add_argument("--index", default=str(INDEX))
    ex = commands.add_parser("export-mobile")
    ex.add_argument("--index", default=str(INDEX))
    ex.add_argument("--out", default=str(ROOT / "models" / "rag_mobile"))
    args = parser.parse_args(argv)

    if args.command == "build":
        from .chunk import chunk_pages
        from .embed import get_embedder
        from .index import Index
        from .load import load_pages
        pages = load_pages(CORPUS)
        chunks = chunk_pages(pages)
        index = Index.build(chunks, get_embedder(args.embedder), out=args.out)
        print(f"{len(pages)} pages -> {len(chunks)} chunks, {index.meta['dim']}-dim {index.meta['embedder']}, saved to {args.out}")
    elif args.command == "ask":
        from .agent import GuidanceAgent
        from .generate import OllamaLLM
        subtype = args.subtype
        if not subtype:
            from ..predict import Predictor
            prediction = Predictor().predict(args.text, task="complaint")["predictions"]["complaint"]
            subtype = prediction["suggested_type"]
            print(f"Predicted subtype: {subtype}", file=sys.stderr)
        agent = GuidanceAgent(args.index, llm=OllamaLLM(args.model) if args.llm == "ollama" else None)
        print(json.dumps(agent.answer(args.text, subtype, args.date), indent=2, ensure_ascii=False))
    elif args.command == "eval":
        from .embed import get_embedder
        from .evaluate import evaluate
        from .index import Index
        index = Index.load(args.index)
        report = evaluate(index, get_embedder(index.meta["embedder"]), out=ROOT / "reports" / "rag_eval.json")
        for name, r in report["results"].items():
            print(f"{name:15s} recall@5={r['recall@5']:.3f}  mrr={r['mrr']:.3f}  (n={r['questions']})")
    else:
        from .index import Index
        from .mobile import export_mobile
        print(f"Exported to {export_mobile(Index.load(args.index), args.out)}")


if __name__ == "__main__":
    main()
