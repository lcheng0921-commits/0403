import numpy as np
import matplotlib.pyplot as plt


def plot(map_data):
    uav_pos = map_data['pos_ubs']
    gts_pos = map_data['pos_gts']
    range_pos = map_data['range_pos']
    area = map_data['area']

    fig, ax = plt.subplots()
    ax.axis([0, range_pos, 0, range_pos])
    for (x, y) in uav_pos:
        ubs, = ax.plot(x, y, marker='o', color='b')
    for (x, y) in gts_pos:
        gts, = ax.plot(x, y, marker='o', color='y', markersize=5)

    for x in range(0, int(range_pos) + 1, int(area)):
        if x == 0:
            continue
        ax.plot([0, range_pos], [x, x], linestyle='--', color='gray', linewidth=0.8)

    for y in range(0, int(range_pos) + 1, int(area)):
        if y == 0:
            continue
        ax.plot([y, y], [0, range_pos], linestyle='--', color='gray', linewidth=0.8)

    ax.legend(handles=[ubs, gts], labels=['uav', 'gts'], loc='upper right')
    ax.set_title('500x500 clustered scenario (MVP)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.show()


class ClusteredMap500:
    """Two-cluster map used by the first MB-PPO implementation round."""

    def __init__(self, range_pos=500, n_eve=4, n_gts=6, n_ubs=1, n_community=4):
        self.n_eve = n_eve
        self.n_gts = n_gts
        self.n_ubs = n_ubs
        self.pos_eve = np.empty((self.n_eve, 2), dtype=np.float32)
        self.pos_gts = np.empty((self.n_gts, 2), dtype=np.float32)
        self.pos_ubs = np.empty((self.n_ubs, 2), dtype=np.float32)
        self.range_pos = range_pos

        # Keep 4 communities to stay compatible with existing security computation.
        self.fen = 2
        self.area = self.range_pos / self.fen
        self.n_community = n_community
        self.gts_in_community = [[] for _ in range(self.n_community)]

    def set_eve(self):
        preset = np.array(
            [
                [80.0, 120.0],
                [420.0, 380.0],
                [-200.0, -200.0],
                [-200.0, -200.0],
            ],
            dtype=np.float32,
        )
        for i in range(self.n_eve):
            if i < len(preset):
                self.pos_eve[i] = preset[i]
            else:
                self.pos_eve[i] = np.array([-200.0, -200.0], dtype=np.float32)

    def set_gts(self):
        self.gts_in_community = [[] for _ in range(self.n_community)]

        if self.n_gts == 3:
            # User-requested deterministic 1+2 split for stable trajectory-style evaluation.
            fixed_points = np.array(
                [
                    [120.0, 345.0],
                    [342.0, 232.0],
                    [382.0, 202.0],
                ],
                dtype=np.float32,
            )
            for gt_idx in range(3):
                point = fixed_points[gt_idx]
                self.pos_gts[gt_idx] = point

                community_id = int((point[1] // self.area) * self.fen + (point[0] // self.area))
                community_id = int(np.clip(community_id, 0, self.n_community - 1))
                self.gts_in_community[community_id].append(gt_idx)
            return
        else:
            n1 = int(np.ceil(self.n_gts * 0.5))
            n2 = int(self.n_gts - n1)
            clusters = [
                (np.array([100.0, 100.0], dtype=np.float32), n1, np.array([25.0, 25.0], dtype=np.float32)),
                (np.array([400.0, 400.0], dtype=np.float32), n2, np.array([25.0, 25.0], dtype=np.float32)),
            ]

        gt_idx = 0
        for center, count, sigma in clusters:
            for _ in range(count):
                if gt_idx >= self.n_gts:
                    break
                point = np.random.normal(loc=center, scale=sigma)
                point = np.clip(point, 20.0, self.range_pos - 20.0)
                self.pos_gts[gt_idx] = point

                community_id = int((point[1] // self.area) * self.fen + (point[0] // self.area))
                community_id = int(np.clip(community_id, 0, self.n_community - 1))
                self.gts_in_community[community_id].append(gt_idx)
                gt_idx += 1

        while gt_idx < self.n_gts:
            point = np.random.uniform(low=20.0, high=self.range_pos - 20.0, size=(2,))
            self.pos_gts[gt_idx] = point
            community_id = int((point[1] // self.area) * self.fen + (point[0] // self.area))
            community_id = int(np.clip(community_id, 0, self.n_community - 1))
            self.gts_in_community[community_id].append(gt_idx)
            gt_idx += 1

    def set_ubs(self):
        if self.n_ubs >= 1:
            self.pos_ubs[0] = np.array([250.0, 250.0], dtype=np.float32)
        for i in range(1, self.n_ubs):
            angle = 2.0 * np.pi * i / self.n_ubs
            offset = 40.0 * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            self.pos_ubs[i] = np.array([250.0, 250.0], dtype=np.float32) + offset

    def get_map(self):
        self.set_eve()
        self.set_gts()
        self.set_ubs()

        return dict(
            pos_gts=self.pos_gts,
            pos_eve=self.pos_eve,
            pos_ubs=self.pos_ubs,
            area=self.area,
            range_pos=self.range_pos,
            gts_in_community=self.gts_in_community,
        )


class ClusteredMap500Split24(ClusteredMap500):
    """Two-cluster map with fixed 2+4 split when n_gts=6."""

    def set_gts(self):
        self.gts_in_community = [[] for _ in range(self.n_community)]

        if self.n_gts != 6:
            # Fallback to parent behavior for non-6-user settings.
            return super().set_gts()

        clusters = [
            (np.array([100.0, 100.0], dtype=np.float32), 2, np.array([25.0, 25.0], dtype=np.float32)),
            (np.array([400.0, 400.0], dtype=np.float32), 4, np.array([25.0, 25.0], dtype=np.float32)),
        ]

        gt_idx = 0
        for center, count, sigma in clusters:
            for _ in range(count):
                if gt_idx >= self.n_gts:
                    break
                point = np.random.normal(loc=center, scale=sigma)
                point = np.clip(point, 20.0, self.range_pos - 20.0)
                self.pos_gts[gt_idx] = point

                community_id = int((point[1] // self.area) * self.fen + (point[0] // self.area))
                community_id = int(np.clip(community_id, 0, self.n_community - 1))
                self.gts_in_community[community_id].append(gt_idx)
                gt_idx += 1


class ClusteredMap500Split42DenseA(ClusteredMap500):
    """Two-cluster map with fixed 4+2 split when n_gts=6.

    Cluster A (dense): center [120, 120], sigma=15, 4 users.
    Cluster B (sparse): center [380, 400], sigma=40, 2 users.
    """

    def set_gts(self):
        self.gts_in_community = [[] for _ in range(self.n_community)]

        if self.n_gts != 6:
            # Fallback to parent behavior for non-6-user settings.
            return super().set_gts()

        clusters = [
            (np.array([120.0, 120.0], dtype=np.float32), 4, np.array([15.0, 15.0], dtype=np.float32)),
            (np.array([380.0, 400.0], dtype=np.float32), 2, np.array([40.0, 40.0], dtype=np.float32)),
        ]

        gt_idx = 0
        for center, count, sigma in clusters:
            for _ in range(count):
                if gt_idx >= self.n_gts:
                    break
                point = np.random.normal(loc=center, scale=sigma)
                point = np.clip(point, 20.0, self.range_pos - 20.0)
                self.pos_gts[gt_idx] = point

                community_id = int((point[1] // self.area) * self.fen + (point[0] // self.area))
                community_id = int(np.clip(community_id, 0, self.n_community - 1))
                self.gts_in_community[community_id].append(gt_idx)
                gt_idx += 1


class ClusteredMap500NearFarExtreme(ClusteredMap500):
    """Extreme Near-Far map with deterministic 3-near + 3-far split for n_gts=6.

    Near users are tightly clustered around UAV initial location (center).
    Far users are placed near three remote corners to induce severe path-loss disparity.
    """

    def set_gts(self):
        self.gts_in_community = [[] for _ in range(self.n_community)]

        if self.n_gts != 6:
            return super().set_gts()

        # 3 near users around center, 3 far users around remote corners.
        clusters = [
            (np.array([250.0, 250.0], dtype=np.float32), 3, np.array([10.0, 10.0], dtype=np.float32)),
            (np.array([460.0, 460.0], dtype=np.float32), 1, np.array([6.0, 6.0], dtype=np.float32)),
            (np.array([460.0, 40.0], dtype=np.float32), 1, np.array([6.0, 6.0], dtype=np.float32)),
            (np.array([40.0, 460.0], dtype=np.float32), 1, np.array([6.0, 6.0], dtype=np.float32)),
        ]

        gt_idx = 0
        for center, count, sigma in clusters:
            for _ in range(count):
                if gt_idx >= self.n_gts:
                    break
                point = np.random.normal(loc=center, scale=sigma)
                point = np.clip(point, 20.0, self.range_pos - 20.0)
                self.pos_gts[gt_idx] = point

                community_id = int((point[1] // self.area) * self.fen + (point[0] // self.area))
                community_id = int(np.clip(community_id, 0, self.n_community - 1))
                self.gts_in_community[community_id].append(gt_idx)
                gt_idx += 1


if __name__ == '__main__':
    map_obj = ClusteredMap500(range_pos=500, n_eve=4, n_gts=6, n_ubs=1, n_community=4)
    map_data = map_obj.get_map()
    plot(map_data)
