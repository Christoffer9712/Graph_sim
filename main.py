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
    start_times=[Time("2026-10-08 14:30:00", scale="utc"),
                 Time("2026-10-10 10:30:00", scale="utc"),
                 Time("2026-10-10 11:30:00", scale="utc"),
                 Time("2026-10-11 5:30:00", scale="utc"),
                 Time("2026-10-13 10:29:00", scale="utc"),
                 Time("2026-10-12 10:15:00", scale="utc"),]
    seed = range(20, 20+len(start_times))
    generate_data(dataset_path, start_times=start_times, seed=seed, analyse=False, plot=False)

model_path = train(dataset_path, device=DEVICE, transfer_learning_model='predictor_models.pth')
print('Training complete. Models saved to', model_path)