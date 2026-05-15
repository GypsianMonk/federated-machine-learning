import torch
import time
from federated.compression import (
    add_state_dicts,
    clone_state_dict,
    deserialize_state_dict,
    serialize_state_dict,
    subtract_state_dicts,
    zero_state_dict_like,
)

class FLClient:
    def __init__(
        self,
        model,
        data,
        labels,
        device="cpu",
        compression_mode="turboquant",
        compression_bits=4,
        batch_size=None,
        class_weights=None,
    ):
        self.model = model.to(device)
        self.X = torch.tensor(data, dtype=torch.float32).to(device)
        self.y = torch.tensor(labels, dtype=torch.long).squeeze().to(device)
        self.device = device
        self.compression_mode = compression_mode
        self.compression_bits = compression_bits
        self.num_examples = int(self.y.shape[0])
        self.error_feedback = None
        self.batch_size = batch_size or self.num_examples
        self.class_weights = None

        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights,
                dtype=torch.float32,
                device=device,
            )

    def train(self, global_state=None, epochs=1, lr=0.001):
        # Limit PyTorch threads per client to avoid GIL/OpenMP thrashing
        # when multiple clients train in parallel via ThreadPoolExecutor
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)

        try:
            if global_state is None:
                global_state = clone_state_dict(self.model.state_dict())

            if global_state is not None:
                self.model.load_state_dict(global_state)

            if self.error_feedback is None:
                self.error_feedback = zero_state_dict_like(global_state)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)

            start_time = time.time()

            self.model.train()
            for _ in range(epochs):
                permutation = torch.randperm(self.num_examples, device=self.device)

                for start in range(0, self.num_examples, self.batch_size):
                    end = start + self.batch_size
                    batch_indices = permutation[start:end]

                    optimizer.zero_grad(set_to_none=True)
                    outputs = self.model(self.X[batch_indices])
                    loss = criterion(outputs, self.y[batch_indices])
                    loss.backward()
                    optimizer.step()

            latency = time.time() - start_time

            # Compression doesn't need gradients
            with torch.no_grad():
                local_state = clone_state_dict(self.model.state_dict())
                delta_state = subtract_state_dicts(local_state, global_state)

                update = {
                    "payload": self._serialize_update(delta_state),
                    "num_examples": self.num_examples,
                    "update_type": "delta",
                }
            return update, latency
        finally:
            torch.set_num_threads(prev_threads)

    def _serialize_update(self, delta_state):
        if self.compression_mode == "none":
            return clone_state_dict(delta_state)

        compensated_delta = add_state_dicts(delta_state, self.error_feedback)
        payload = serialize_state_dict(
            compensated_delta,
            compression_mode=self.compression_mode,
            compression_bits=self.compression_bits,
        )
        restored_delta = deserialize_state_dict(payload)
        self.error_feedback = subtract_state_dicts(compensated_delta, restored_delta)
        return payload