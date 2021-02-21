import heapq
from typing import Dict
from typing import List
from typing import Tuple

from pyclam import Cluster
from pyclam import Manifold
from pyclam import criterion

from src.toy_shapes import SHAPES
from src.utils import *

# Cluster -> (birth-radius, death-radius)
Barcodes = Dict[Cluster, Tuple[float, float]]


class Code:
    def __init__(self, cluster: Cluster, death: float, birth: float = -1):
        self._cluster: Cluster = cluster
        self._death: float = death
        self._birth: float = birth
        return

    def set_birth(self, birth: float):
        self._birth = birth
        return

    @property
    def cluster(self) -> Cluster:
        return self._cluster

    @property
    def death(self) -> float:
        return self._death

    @property
    def birth(self) -> float:
        return self._birth

    @property
    def radius(self) -> float:
        return self._cluster.radius

    def __lt__(self, other: 'Code'):
        return self.cluster.radius > other.cluster.radius


def _normalize(factor: float, barcodes: Barcodes) -> Barcodes:
    return {c: (b / factor, d / factor) for c, (b, d) in barcodes.items()}


def _group_by_cardinality(barcodes: Barcodes) -> Dict[int, Barcodes]:
    cardinalities: List[int] = list(sorted({cluster.cardinality for cluster in barcodes}))
    barcodes_by_cardinality: Dict[int, Barcodes] = {cardinality: dict() for cardinality in cardinalities}
    [barcodes_by_cardinality[cluster.cardinality].update({cluster: lifetime})
     for cluster, lifetime in barcodes.items()]
    return barcodes_by_cardinality


def _merge_high_cardinalities(
        threshold: int,
        barcodes_by_cardinality: Dict[int, Barcodes],
) -> Dict[int, Barcodes]:
    # Merges all barcodes for clusters with cardinality greater than 'threshold'
    high_cardinalities = [v for c, v in barcodes_by_cardinality.items() if c >= threshold]
    if len(high_cardinalities) > 0:
        [high_cardinalities[0].update(h) for h in high_cardinalities[1:]]
        barcodes_by_cardinality = {c: v for c, v in barcodes_by_cardinality.items() if c < threshold}
        barcodes_by_cardinality[threshold] = high_cardinalities[0]
    return barcodes_by_cardinality


def create_barcodes(
        data: np.array,
        *,
        normalize: bool = True,
        merge: Optional[int] = 4,
) -> Dict[int, Barcodes]:
    manifold: Manifold = Manifold(data, 'euclidean').build_tree(criterion.MaxDepth(20))
    barcodes: Barcodes = dict()

    # living-clusters is a heap with highest radius at the top
    living_clusters = [Code(manifold.root, manifold.root.radius)]
    heapq.heapify(living_clusters)

    while living_clusters:  # Go over max-heap
        current: Code = heapq.heappop(living_clusters)

        if current.cluster.children:  # handle children
            current.set_birth(current.radius)
            [left, right] = list(current.cluster.children)

            if left.radius >= current.radius:  # left is still-born
                barcodes[left] = (current.radius, current.radius)
            else:  # or added to living clusters
                heapq.heappush(living_clusters, Code(left, current.radius))

            if right.radius >= current.radius:  # right is still-born
                barcodes[right] = (current.radius, current.radius)
            else:  # or added to living-clusters
                heapq.heappush(living_clusters, Code(right, current.radius))

        else:  # otherwise set birth to zero-radius
            current.set_birth(0.)
        # add current to dict of barcodes
        barcodes[current.cluster] = (current.birth, current.death)

    if normalize:
        barcodes = _normalize(manifold.root.radius, barcodes)

    barcodes_by_cardinality = _group_by_cardinality(barcodes)

    if merge is not None:
        barcodes_by_cardinality = _merge_high_cardinalities(merge, barcodes_by_cardinality)

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
                normalize=True,
                merge=4,
            )
            cardinalities = list(sorted(list(barcodes_by_cardinality.keys())))
            for cardinality in cardinalities:
                barcodes = barcodes_by_cardinality[cardinality]
                barcodes = list(sorted([(birth, death) for birth, death in barcodes.values()]))
                with open(filename, 'a') as fp:
                    for birth, death in barcodes:
                        fp.write(f'{cardinality},{birth:.12f},{death:.12f}\n')
    return


def plot(num_points: int = 10**2):
    report_file = os.path.join(PLOTS_DIR, 'time.txt')
    if os.path.exists(report_file):
        os.remove(report_file)
    for shape in SHAPES:
        barcodes = create_barcodes(
            SHAPES[shape](num_points=num_points).T,
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
