import argparse

from baby_whale_v4.cli import bench, data, eval, generate, serve, smoke, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="baby-whale-v4")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke.register(sub)
    train.register(sub)
    serve.register(sub)
    generate.register(sub)
    eval.register(sub)
    data.register(sub)
    bench.register(sub)

    args = parser.parse_args(argv)
    args.func(args)


__all__ = ["main"]


if __name__ == "__main__":
    main()
