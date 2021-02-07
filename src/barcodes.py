from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

from pyclam import Cluster
from pyclam import Manifold

from src.toy_shapes import SHAPES
from src.utils import *

# Cluster -> (birth-radius, death-radius)
Barcodes = Dict[Cluster, Tuple[float, float]]


def create_barcodes(
        data: np.array,
        *,
        steps: Optional[int] = 10**3,
        normalize: bool = False,
        merge: Optional[int] = None,
) -> Dict[int, Barcodes]:
    manifold: Manifold = Manifold(data, 'euclidean').build()
    thresholds: np.array = np.linspace(start=manifold.root.radius * (steps - 1) / steps, stop=0, num=steps)
    barcodes: Barcodes = dict()
    living_clusters: Barcodes = {manifold.root: (-1, manifold.root.radius)}
    for threshold in thresholds:
        new_births: Set[Cluster] = set()
        dead_clusters: Set[Cluster] = {cluster for cluster in living_clusters if cluster.radius > threshold}
        while dead_clusters:
            cluster = dead_clusters.pop()
            death = living_clusters.pop(cluster)[1] if cluster in living_clusters else threshold
            barcodes[cluster] = threshold, death
            for child in cluster.children:
                if child.cardinality > 1:
                    (dead_clusters if child.radius > threshold else new_births).add(child)
                else:
                    barcodes[child] = (0, threshold)
        living_clusters.update({cluster: (-1, threshold) for cluster in new_births})

    if normalize:  # normalize radii to [0, 1] range.
        factor = manifold.root.radius
        barcodes = {c: (b / factor, d / factor) for c, (b, d) in barcodes.items()}
    
    cardinalities: List[int] = list(sorted({cluster.cardinality for cluster in barcodes}))
    barcodes_by_cardinality: Dict[int, Barcodes] = {cardinality: dict() for cardinality in cardinalities}
    [barcodes_by_cardinality[cluster.cardinality].update({cluster: lifetime})
     for cluster, lifetime in barcodes.items()]
    
    if merge is not None:
        # Merges all barcodes for clusters with cardinality greater than 'merge'
        high_cardinalities = [v for c, v in barcodes_by_cardinality.items() if c >= merge]
        if len(high_cardinalities) > 0:
            [high_cardinalities[0].update(h) for h in high_cardinalities[1:]]
            barcodes_by_cardinality = {c: v for c, v in barcodes_by_cardinality.items() if c < merge}
            barcodes_by_cardinality[merge] = high_cardinalities[0]
    
    return barcodes_by_cardinality


def plot_barcodes(barcodes_by_cardinality: Dict[int, Barcodes], filename: str):
    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_subplot(111)

    height = 1
    y_ticks: Dict[int, str] = dict()
    for cardinality, barcodes in barcodes_by_cardinality.items():
        lifetimes = list(sorted(barcodes.values()))
        for birth, death in lifetimes:
            plt.plot([birth, death], [height, height], lw=0.8)
            height += 1
        y_ticks[height] = f'{cardinality}'

    ax.set_yticks(list(y_ticks.keys()))
    ax.set_yticklabels(list(y_ticks.values()))

    plt.xlabel('radius')
    plt.ylabel('cardinality')
    plt.title('barcodes')
    plt.savefig(filename, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)
    return


def write(
        num_points: int = 10**2,
        resolution: int = 10**2,
        number_per_shape: int = 10**1,
):
    for shape in SHAPES:
        filename = os.path.join(BARCODES_DIR, f'{shape}__0.csv')
        for _ in range(number_per_shape):
            filename = increment_filename(filename)
            with open(filename, 'w') as fp:
                fp.write('cardinality,birth,death\n')

            barcodes_by_cardinality = create_barcodes(
                SHAPES[shape](num_points=num_points).T,
                steps=resolution,
                normalize=True,
                merge=4,
            )
            cardinalities = list(sorted(list(barcodes_by_cardinality.keys())))
            for cardinality in cardinalities:
                barcodes = barcodes_by_cardinality[cardinality]
                barcodes = list(sorted([(birth, death) for birth, death in barcodes.values()]))
                with open(filename, 'a') as fp:
                    for birth, death in barcodes:
                        fp.write(f'{cardinality},{birth:.4f},{death:.4f}\n')
    return


def plot(num_points: int = 10**2, resolution: int = 10**2):
    report_file = os.path.join(PLOTS_DIR, 'time.txt')
    if os.path.exists(report_file):
        os.remove(report_file)
    for shape in SHAPES:
        barcodes = create_barcodes(
            SHAPES[shape](num_points=num_points).T,
            steps=resolution,
            normalize=True,
            merge=4,
        )
        plot_barcodes(
            barcodes,
            os.path.join(PLOTS_DIR, f'barcodes-{shape}.png'),
        )
    return


if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(BARCODES_DIR, exist_ok=True)
    # plot()
    write()
