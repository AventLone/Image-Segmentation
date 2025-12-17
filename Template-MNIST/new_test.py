from tqdm import tqdm
import time

n_train = 1000
epochs = 5

for epoch in range(epochs):
    with tqdm(total=n_train,
              desc=f"Epoch {epoch + 1}/{epochs}",
              unit="img") as pbar:

        for i in range(n_train):
            # do your training step here
            loss = 0.0234
            time.sleep(0.01)

            # update progress bar
            pbar.update(1)
            # optional: show extra info
            pbar.set_postfix(loss=f"{loss:.4f}")