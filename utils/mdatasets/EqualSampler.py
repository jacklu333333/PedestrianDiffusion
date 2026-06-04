from .common_imports import *
from .utils import *


class EqualSampler(Sampler):
    def __init__(self, datasets, num_samples_per_dataset, mode):
        self.datasets = datasets
        self.num_samples_per_dataset = num_samples_per_dataset
        self.mode = mode
        # Calculate the offset for each dataset
        self.offsets = []
        current_offset = 0
        for d in datasets:
            self.offsets.append(current_offset)
            current_offset += len(d)

        # add counter
        self.counter = []
        for dataset in datasets:
            self.counter.append(torch.zeros(len(dataset), dtype=torch.int))

    def __iter__(self):
        # sample base on the counter, once sample the index of the counter add by 1, the higher the counter lower possibility to sample this index
        if self.mode == "train":
            indices = []
            for dataset_idx, (offset, dataset, counter) in enumerate(
                zip(self.offsets, self.datasets, self.counter)
            ):
                # Calculate sampling weights inversely proportional to the counter (add 1 to avoid zero division)
                weights = 1.0 / (counter + 1).float()
                weights = weights / weights.sum()
                sampled_indices = torch.multinomial(
                    weights, self.num_samples_per_dataset, replacement=True
                )
                # Update the counter for sampled indices
                for idx in sampled_indices:
                    counter[idx] += 1
                indices.extend([offset + idx.item() for idx in sampled_indices])
            random.shuffle(indices)
            return iter(indices)
        else:
            indices = []
            for offset, dataset in zip(self.offsets, self.datasets):
                indices.extend(range(offset, offset + self.num_samples_per_dataset))
            # Interleave indices from each dataset
            new_indices = []
            for i in range(self.num_samples_per_dataset):
                for j in range(len(self.datasets)):
                    idx = j * self.num_samples_per_dataset + i
                    if idx < len(indices):
                        new_indices.append(indices[idx])
            return iter(new_indices)

    # def __iter__(self):
    #     if self.mode == "train":
    #         indices = []
    #         for offset, dataset in zip(self.offsets, self.datasets):
    #             sampled = random.choices(
    #                 range(len(dataset)), k=self.num_samples_per_dataset
    #             )
    #             indices.extend([offset + i for i in sampled])
    #         random.shuffle(indices)
    #         return iter(indices)
    #     else:
    #         # get the first self.num_samples_per_dataset samples from each dataset
    #         indices = []
    #         for offset, dataset in zip(self.offsets, self.datasets):
    #             indices.extend(range(offset, offset + self.num_samples_per_dataset))
    #         # reorder the datasets: ex if there is 5  datasets, then pick the each of them from the datasets
    #         new_indices = []
    #         for i in range(self.num_samples_per_dataset):
    #             for j in range(len(self.datasets)):
    #                 if i + j * self.num_samples_per_dataset < len(indices):
    #                     new_indices.append(
    #                         indices[i + j * self.num_samples_per_dataset]
    #                     )
    #         indices = new_indices
    #         return iter(indices)

    def __len__(self):
        return self.num_samples_per_dataset * len(self.datasets)
