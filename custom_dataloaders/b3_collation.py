from torch.utils.data import default_collate

def custom_collate_fn(batch):
    frame, frame_class, player_images, player_labels = default_collate(batch)
    return player_images, frame_class
