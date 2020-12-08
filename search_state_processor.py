import argparse
import csv
import pickle

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.colors import to_rgb


def compute_x_ticks(x_min, x_max):
    minor_ticks = np.concatenate([np.linspace(0.0, 0.1, 11), np.linspace(0.2, 1.0, 17)])
    x_minor_ticks = minor_ticks[np.logical_and(minor_ticks >= x_min, minor_ticks <= x_max)]
    major_ticks = np.concatenate(
        [[x_min] if x_max >= 0.2 else np.linspace(0.0, 0.1, 6),
         np.linspace(0.1, max(x_max, 0.1), round((x_max - 0.1) / 0.05) + 1)])
    x_major_ticks = major_ticks[np.logical_and(major_ticks >= x_min, major_ticks <= x_max)]
    return x_minor_ticks, x_major_ticks


def is_pareto_efficient(points):
    points = np.asarray(points)
    is_efficient = np.ones(points.shape[0], dtype=np.bool)
    for i, c in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(points[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


def get_pareto_front(points):
    eff_mask = is_pareto_efficient(points)
    return [p for p, e in zip(points, eff_mask) if e]


def plot_pareto_front(search_state_file, x_range=(0.0, 1.0), y_range=(0.0, 3e6), title=None,
                      output_file=None):
    points = load_search_state_file(search_state_file)
    error = np.array([o[0] for o in points])
    pmu = np.array([o[1] for o in points])
    ms = np.array([o[2] for o in points])
    macs = np.array([o[3] for o in points])
    is_efficient = is_pareto_efficient(points)

    x_min, x_max = x_range
    y_min, y_max = y_range
    x_minor_ticks, x_major_ticks = compute_x_ticks(x_min, x_max)

    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=[3.8, 3.2], dpi=300)
    ax = fig.add_subplot()

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_xticks(x_major_ticks, minor=False)
    ax.set_xlabel("Error rate")
    ax.set_ylabel("Resource usage (bytes)")
    ax.set_yscale("log")
    ax.set_title(title or "PMU, model size and MACs versus error rate")
    ax.xaxis.grid(True, which='both', linewidth=0.5, linestyle=":")
    ax.yaxis.grid(True, which='major', linewidth=0.5, linestyle=":")

    colors = [f"C{i}" for i in range(3)]

    def scatter(x, y, color, alpha, label):
        r, g, b = to_rgb(color)
        color = [(r, g, b, a) for a in alpha]
        ax.scatter(x, y, marker="D", s=10, label=label, color=color)

    # scatter(error, macs, color=colors[0], label="MACs",
    #         alpha=(0.1 + 0.25 * is_efficient))
    scatter(error, pmu, color=colors[1], label="Peak memory usage",
            alpha=(0.1 + 0.25 * is_efficient))
    # plt.hlines(np.mean(pmu), x_min, x_max, color=colors[1])
    scatter(error, ms, color=colors[2], label="Model size",
            alpha=(0.1 + 0.25 * is_efficient))
    # plt.hlines(np.mean(ms), x_min, x_max, color=colors[2])
    ax.legend(loc="upper right")

    for i, c in enumerate(colors[1:]):
        ax.legend_.legendHandles[i].set_facecolor(c)

    plt.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=fig.dpi)
    else:
        plt.show()


def output_csv(objectives):
    print("Error,PMU,MS,MACs")
    for obj in objectives:
        print(f"{obj[0]:.4f},{obj[1]},{obj[2]},{obj[3]}")


def load_search_state_file(search_state_file, filter_resources=None):
    is_bo = "agingevosearch" not in search_state_file
    with open(search_state_file, "rb") as f:
        if is_bo:
            wrapped_nns = [nn[0] for nn in pickle.load(f)["points"]]
        else:
            wrapped_nns = pickle.load(f)

    points = [(nn.test_error, nn.resource_features[0],
               nn.resource_features[1], nn.resource_features[2])
              for nn in wrapped_nns]

    if filter_resources:
        key = filter_resources
        points = [(o[0], o[key]) for o in points]

    return points


def plot_accuracy_gain(search_state_file, x_range=(100, 2000), y_range=(0.8, 1.0),
                       ms_filter=64_000, output_file=None):
    points = load_search_state_file(search_state_file)
    max_accuracies = np.maximum.accumulate([1.0 - o[0] for o in points])
    steps = np.arange(1, len(max_accuracies) + 1)

    x, y = [], []
    for p, max_acc, step in zip(points, max_accuracies, steps):
        if p[2] > ms_filter:
            continue
        x.append(step)
        y.append(max_acc)

    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=[3.0, 5.4], dpi=300)
    ax = fig.add_subplot()

    x_min, x_max = x_range
    y_min, y_max = y_range

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    ax.step(x, y, where="post")
    ax.xaxis.grid(True, which='both', linewidth=0.5, linestyle=":")
    ax.yaxis.grid(True, which='major', linewidth=0.5, linestyle=":")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Accuracy")

    plt.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=fig.dpi)
    else:
        plt.show()


def multiple_pareto_fronts(search_state_files, descriptions, y_key=2, take_n=2000,
                           x_range=(0.0, 1.0), y_range=(0.0, 3e6), title=None, output_file=None):
    point_lists = [load_search_state_file(file, filter_resources=y_key)[:take_n]
                   for file in search_state_files]

    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=[5.4, 3.0], dpi=300)
    ax = fig.add_subplot()

    x_min, x_max = x_range
    x_minor_ticks, x_major_ticks = compute_x_ticks(x_min, x_max)
    y_min, y_max = y_range

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_xticks(x_major_ticks, minor=False)

    colors = [f"C{i}" for i in range(len(point_lists))]

    def scatter(x, y, color, alpha, label):
        r, g, b = to_rgb(color)
        color = [(r, g, b, a) for a in alpha]
        ax.scatter(x, y, marker="D", s=10, label=label, color=color)

    for points, desc, color in zip(point_lists, descriptions, colors):
        points.sort(key=lambda x: x[0])
        is_eff = is_pareto_efficient(points)
        err = np.array([o[0] for o in points])
        res = np.array([o[1] for o in points])
        scatter(err, res, label=desc, alpha=(0.04 + 0.96 * is_eff), color=color)
        ax.step(err[is_eff], res[is_eff], where="post", alpha=0.7)

    ax.xaxis.grid(True, which='both', linewidth=0.5, linestyle=":")
    ax.yaxis.grid(True, which='major', linewidth=0.5, linestyle=":")
    ax.set_xlabel("Error rate")
    ax.set_ylabel(["Error rate", "Peak memory usage", "Model size", "MACs"][y_key])
    if title:
        ax.set_title(title)

    ax.legend()
    for i, c in enumerate(colors):
        ax.legend_.legendHandles[i].set_facecolor(c)

    plt.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=fig.dpi)
    else:
        plt.show()


def process(output_mode, search_state_file):
    objs = load_search_state_file(search_state_file)
    if output_mode == "csv":
        output_csv(objs)
    elif output_mode == "pareto_plot":
        plot_pareto_front(objs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output_mode", type=str, choices=["csv", "pareto_plot"])
    p.add_argument("search_state_file", type=str)
    args = p.parse_args()

    process(args.output_mode, args.search_state_file)


def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results['polynomial'] = coeffs

    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y) / len(y)
    ssreg = np.sum((yhat - ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r2'] = ssreg / sstot

    return results


def plot_latency_vs_mac(latency_file, take_n=1000, x_range=(0, 90), y_range=(0, 800),
                        output_file=None):
    data = defaultdict(list)
    with open(latency_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except ValueError:
                    data[k].append(v)

    latency = np.array(data["latency"][:take_n]) / 1000  # Convert to ms
    macs = np.array(data["MACs"][:take_n]) / 1_000_000  # Convert to M-MACs

    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=[4.0, 2.5], dpi=300)
    ax = fig.add_subplot()

    ax.set_xlim(list(x_range))
    ax.set_ylim(list(y_range))

    ax.xaxis.grid(True, which='both', linewidth=0.5, linestyle=":")
    ax.yaxis.grid(True, which='major', linewidth=0.5, linestyle=":")
    ax.set_xlabel("MACs (M)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Measured MCU latency vs MACs")
    ax.scatter(macs, latency, s=5, color="C0", alpha=0.7)

    # Plot trendline
    z = polyfit(macs, latency, 1)
    p = np.poly1d(z["polynomial"])
    ax.plot(macs, p(macs), "-", linewidth=0.8, color="C1")
    box_props = \
        {'ha': 'center', 'va': 'center'}
    ax.text(70, 400, f"$R^2 = {z['r2']:.3f}$", box_props, rotation=25)

    plt.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=fig.dpi)
    else:
        plt.show()


if __name__ == '__main__':
    main()
    # plot_latency_vs_mac("artifacts/latency.csv", output_file="mcu_latency.pdf")
    # plot_pareto_front("artifacts/cnn_mnist/no_ms_agingevosearch_state.pickle",
    #                   x_range=(0.00, 0.04), y_range=(100, 3_000_000),
    #                   title="μNAS on MNIST w/o model size constraint",
    #                   output_file="mnist_no_ms.pdf")
    # plot_pareto_front("artifacts/cnn_mnist/no_pmu_agingevosearch_state.pickle",
    #                   x_range=(0.00, 0.04), y_range=(100, 3_000_000),
    #                   title="μNAS on MNIST w/o mem. usage constraint",
    #                   output_file="mnist_no_pmu.pdf")
    # plot_pareto_front("artifacts/cnn_mnist/plain_final_agingevosearch_state.pickle",
    #                   x_range=(0.00, 0.04), y_range=(100, 3_000_000),
    #                   title="μNAS on MNIST w/ all constraints",
    #                   output_file="mnist_all.pdf")
    # multiple_pareto_fronts(
    #     ["artifacts/cnn_mnist/plain_final_agingevosearch_state.pickle",
    #      "artifacts/cnn_mnist/plain_final_bo_search_state.pickle",
    #      "artifacts/cnn_mnist/new_struct_agingevosearch_state.pickle"],
    #     ["Aging Evo. (AE)", "Bayes. Opt. (BO)", "AE + Pruning"],
    #     x_range=(0.0, 0.1), y_range=(0, 8000), y_key=2,
    #     title="Model size vs error rate Pareto fronts for MNIST",
    #     output_file="pareto_mnist.pdf")
    # multiple_pareto_fronts(
    #     ["artifacts/cnn_chars74k/plain_final_agingevosearch_state.pickle",
    #      "artifacts/cnn_chars74k/bo_plain_final_bo_search_state.pickle",
    #      "artifacts/cnn_chars74k/struct_final_agingevosearch_state.pickle"],
    #     ["Aging Evo. (AE)", "Bayes. Opt. (BO)", "AE + Pruning"],
    #     x_range=(0.10, 0.50), y_range=(0, 30000), y_key=2,
    #     title="Model size vs error rate Pareto fronts for Chars74K",
    #     output_file="pareto_chars74k.pdf")
    #
