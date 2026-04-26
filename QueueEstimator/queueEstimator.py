import time
import datetime
import csv
import os

class QueueEstimator:
    def __init__(self, max_fov=5, discharge_rate=0.5, csv_path="historical_traffic.csv"):

        self.MAX_FOV = max_fov
        self.DISCHARGE_RATE = discharge_rate
        
        self.hidden_queue = 0.0
        self.last_update_time = time.time()
        
        self.temporal_data = self._load_historical_data(csv_path)

    def _load_historical_data(self, path):
        data = {}
        if not os.path.exists(path):
            print(f"csv not found")
            return data

        with open(path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                day = row["day_of_week"]
                hour = int(row["hour"])
                rate = float(row["arrival_rate_per_sec"])
                data[(day, hour)] = rate
                
        return data

    def get_dynamic_arrival_rate(self):
        now = datetime.datetime.now()
        current_day = now.strftime("%A")
        current_hour = now.hour        

        rate = self.temporal_data.get((current_day, current_hour), 0.2) 
        return rate

    def update(self, current_time, detected_cars, light_state):
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        is_saturated = (detected_cars >= self.MAX_FOV)

        if is_saturated and light_state == "RED":
            current_arrival_rate = self.get_dynamic_arrival_rate()
            self.hidden_queue += (current_arrival_rate * dt)

        elif light_state == "GREEN":
            self.hidden_queue -= (self.DISCHARGE_RATE * dt)
            if self.hidden_queue < 0:
                self.hidden_queue = 0.0

        elif not is_saturated and light_state == "RED":
            self.hidden_queue = 0.0

        true_queue = detected_cars + int(self.hidden_queue)
        return true_queue, int(self.hidden_queue)
 
# USAGE
# create the object, you neeed to create separate object for each lane to track them independently
# estimator = QueueEstimator()o
# in your loop, call update with the current time, detected cars, and light state
# true_queue, predicted_hidden = estimator.update(time.time(), detected_cars, light_state)
# then use the  true quee to feed the agent make sure to normalize