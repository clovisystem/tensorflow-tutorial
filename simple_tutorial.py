import numpy as np
import math

def normalize(array):
    return (array - array.mean()) / array.std()

# Generate some house sizes between 1000 and 3500 (typical sq ft of houses)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=90, high=325, size=num_house)

# Generate house prices from the sizes with a random noise added
np.random.seed(42)
house_price = house_size * 1000.0 + np.random.randint(low=20000, high=70000, size=num_house)
num_train_samples = math.floor(num_house * 0.7)

# Training data (70%)
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples])
train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

#Test data( 30%)
test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asarray(house_price[num_train_samples:])    

