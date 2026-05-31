from dataclasses import dataclass, field

import numpy as np

from utils.data_split import split_iid_data


@dataclass
class PhaseSchedule:
    T_drift: float
    num_phases: int
    active_groups: list = field(default_factory=list)
    drift_times: list = field(default_factory=list)

    @classmethod
    def create(cls, T_drift, num_phases, group_A, group_B):
        active_groups = []
        for p in range(num_phases):
            active_groups.append(group_A if p % 2 == 0 else group_B)
        drift_times = [T_drift * (i + 1) for i in range(num_phases - 1)]
        return cls(
            T_drift=T_drift,
            num_phases=num_phases,
            active_groups=active_groups,
            drift_times=drift_times,
        )

    @property
    def total_time(self):
        return self.T_drift * self.num_phases

    def get_phase_at(self, virtual_time):
        if virtual_time < 0:
            return 0
        phase = int(virtual_time // self.T_drift)
        return min(phase, self.num_phases - 1)

    def get_active_classes(self, phase):
        if phase < 0 or phase >= self.num_phases:
            return []
        return self.active_groups[phase]

    '''
    Dado uma lista de grupos, exemplo: [[0,1,2,3,4], [5,6,7,8,9], [0,1,2,3,4]]
    Indexa cada grupo dessa lista a uma letra, exemplo: A, B, C, ...
    '''
    def get_group_masks(self):
        seen_groups = {}
        masks = {}
        label = ord("A")
        for group in self.active_groups:
            key = tuple(group)
            if key not in seen_groups:
                # (0, 1, 2, 3, 4): A
                seen_groups[key] = chr(label)
                label += 1
            # A: [0, 1, 2, 3, 4]
            masks[seen_groups[key]] = list(group)
        return masks


def filter_by_classes(x, y, active_classes):
    mask = np.isin(y, active_classes)
    return x[mask], y[mask], np.where(mask)[0]


def precompute_phase_indices(y_train, num_clients, schedule):
    phase_indices = []
    for phase in range(schedule.num_phases):
        active_classes = schedule.get_active_classes(phase)
        _, y_filtered, original_indices = filter_by_classes(
            np.arange(len(y_train)), y_train, active_classes
        )
        client_splits = split_iid_data(
            (original_indices, y_filtered),
            num_clients,
            len(active_classes),
        )
        phase_indices.append(client_splits)
    return phase_indices
