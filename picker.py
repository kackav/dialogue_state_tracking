import yaml
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument("--metric", default="jga")
    parser.add_argument("--dataset", default="multiwoz")
    parser.add_argument("input_metric_yamls", nargs="+")

    args = parser.parse_args()

    best_metric = float('inf') if args.minimize else -float('inf')
    best_checkpoint = None

    for yaml_file in args.input_metric_yamls:
        d = yaml.load(open(yaml_file), yaml.SafeLoader)
        d2 = [_ for _ in d if _["dataset"] == args.dataset]
        
        metric = d2[-1][args.metric]
        if (args.minimize and metric < best_metric) or (not args.minimize and metric > best_metric):
            best_metric = metric
            best_checkpoint = os.path.dirname(yaml_file)

    print(best_checkpoint)


if __name__ == '__main__':
    main()