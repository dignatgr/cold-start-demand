"""Command line entry point: ``python -m coldstart.run --raw-dir <H&M folder> --out results``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from coldstart.config import Config
from coldstart.data import build_new_product_dataset, load_raw
from coldstart.evaluate import (
    format_deltas,
    format_pooled,
    metrics_by_origin,
    paired_deltas,
    plot_wape,
    pooled_metrics,
    to_markdown,
)
from coldstart.experiment import run_experiment
from coldstart.features import align_embeddings
from coldstart.splits import rolling_origin_splits
from coldstart.synthetic import make_synthetic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-dir", type=Path, help="Folder with articles.csv, transactions_train.csv, images/"
    )
    p.add_argument("--synthetic", action="store_true", help="Use a small synthetic dataset instead")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--cache", type=Path, default=Path("cache"), help="Embedding cache folder")
    p.add_argument(
        "--max-products", type=int, default=None, help="Subsample products (quick tests)"
    )
    p.add_argument("--no-mlp", action="store_true", help="Skip the PyTorch MLP")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=2, help="Image loading workers")
    args = p.parse_args(argv)
    if not args.synthetic and args.raw_dir is None:
        p.error("--raw-dir is required unless --synthetic is set")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = Config(n_boot=args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        tx, articles, ids, emb = make_synthetic()
        df = build_new_product_dataset(tx, articles, cfg)
    else:
        tx, articles = load_raw(args.raw_dir)
        df = build_new_product_dataset(tx, articles, cfg)
        del tx
        if args.max_products:
            codes = pd.Series(df["product_code"].unique()).sample(
                args.max_products, random_state=cfg.seed
            )
            df = df[df["product_code"].isin(codes)].reset_index(drop=True)
        from coldstart.embeddings import compute_embeddings  # needs PyTorch

        ids, emb = compute_embeddings(
            df["article_id"].to_numpy(),
            args.raw_dir / "images",
            args.cache,
            num_workers=args.num_workers,
        )
    df, emb = align_embeddings(df, ids, emb)
    print(
        f"{len(df)} new articles with an image, launches "
        f"{df.launch_date.min().date()} to {df.launch_date.max().date()}"
    )

    preds = run_experiment(df, emb, cfg, use_mlp=not args.no_mlp)

    splits = pd.DataFrame(
        [
            {
                "origin": s.origin,
                "cutoff": s.cutoff.date(),
                "n_fit": int(s.fit.sum()),
                "n_val": int(s.val.sum()),
                "n_test": int(s.test.sum()),
            }
            for s in rolling_origin_splits(df, cfg)
        ]
    )
    by_origin = metrics_by_origin(preds)
    pooled = pooled_metrics(preds, cfg)
    deltas = paired_deltas(preds, cfg)

    preds.to_csv(args.out / "predictions.csv.gz", index=False)
    splits.to_csv(args.out / "splits.csv", index=False)
    by_origin.to_csv(args.out / "metrics_by_origin.csv", index=False)
    pooled.to_csv(args.out / "metrics_pooled.csv", index=False)
    deltas.to_csv(args.out / "paired_deltas.csv", index=False)
    plot_wape(pooled, args.out / "wape_pooled.png")

    report = "\n\n".join(
        [
            f"Articles: {len(df)}; origins: {len(splits)}",
            to_markdown(splits),
            "Pooled test metrics (95% intervals from a bootstrap over products):",
            to_markdown(format_pooled(pooled)),
            "Paired differences in WAPE (negative: the first model is better):",
            to_markdown(format_deltas(deltas)),
        ]
    )
    (args.out / "results.md").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
