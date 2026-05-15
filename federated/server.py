from federated.fedavg import fed_avg
from federated.compression import (
    deserialize_state_dict,
    is_serialized_payload,
    summarize_payloads,
)

class FLServer:
    def __init__(self, model):
        self.global_model = model
        self.last_compression_stats = None

    def aggregate(self, client_updates):
        decoded_updates = []
        client_weights = []
        update_type = "weights"
        payloads = []

        for update in client_updates:
            if isinstance(update, dict) and "payload" in update:
                payload = update["payload"]
                client_weights.append(float(update.get("num_examples", 1)))
                update_type = update.get("update_type", update_type)
                payloads.append(payload)

                if is_serialized_payload(payload):
                    decoded_updates.append(deserialize_state_dict(payload))
                else:
                    decoded_updates.append(payload)
            elif is_serialized_payload(update):
                payloads.append(update)
                client_weights.append(1.0)
                decoded_updates.append(deserialize_state_dict(update))
            else:
                payloads.append(update)
                client_weights.append(1.0)
                decoded_updates.append(update)

        self.last_compression_stats = summarize_payloads(payloads)
        self.global_model = fed_avg(
            self.global_model,
            decoded_updates,
            client_weights=client_weights,
            update_type=update_type,
        )
        return self.global_model
