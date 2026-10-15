import os
from generate_data import generate_data
from astropy.time import Time
from trainer.train_predictor import train

RECALCULATE_DATA = True
DEVICE = 'cpu'
#----------------------------------------
dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'predictor_dataset.pt')
dataset_path = os.path.abspath(dataset_path)

# Ensure data dir exists
data_dir = os.path.dirname(dataset_path)
os.makedirs(data_dir, exist_ok=True)

if RECALCULATE_DATA:
    start_times=[Time("2026-10-08 12:30:00", scale="utc"),
                 Time("2026-10-10 09:30:00", scale="utc"),
                 Time("2026-10-10 11:23:00", scale="utc"),
                 Time("2026-10-11 05:34:00", scale="utc"),
                 Time("2026-10-13 10:49:00", scale="utc"),
                 Time("2026-10-12 10:12:00", scale="utc"),]
    seed = range(50, 50+len(start_times))
    generate_data(dataset_path, start_times=start_times, seed=seed, analyse=False, plot=True)

model_path = train(dataset_path, device=DEVICE, transfer_learning_model='predictor_models.pth')
print('Training complete. Models saved to', model_path)