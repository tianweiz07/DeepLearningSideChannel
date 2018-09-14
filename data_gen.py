import time
import random
from datetime import datetime
import mxnet as mx
from mxnet import nd

random.seed(datetime.now())

# speed: how many victim's accesses when finishing one prime
# coverage: the ratio that attacker can cover the cache

def GenData(num_batches, batch_size, num_sets, speed=1, coverage=1):

    cache_state = [0]*num_sets
    boundary = num_sets/coverage
    interval = boundary/speed

    x = nd.zeros((num_batches, batch_size, num_sets))
    y = nd.zeros((num_batches, batch_size, num_sets))
    for i in range(num_batches):
        for j in range(batch_size):
            for k in range(speed):
                index = random.randint(0, num_sets-1)
                while y[i, j, index].asscalar() > 0:
                    index = random.randint(0, num_sets-1)

                y[i, j, index] = k+1
                cache_state[index] = 1

                for l in range(k*interval, k*interval+interval):
                    if cache_state[l] == 1:
                        x[i, j, l] = 1
                        cache_state[l] = 0

    return x, y

# This generates the dataset when there are victim access
# between the Set and Check operations.

def GenDataInterval(num_batches, batch_size, num_sets, pinterval=0, speed=1, coverage=1):

    cache_state = [0]*num_sets
    boundary = num_sets/coverage
    interval = boundary/speed

    x = nd.zeros((num_batches, batch_size, num_sets))
    y = nd.zeros((num_batches, batch_size, num_sets))
    for i in range(num_batches):
        for j in range(batch_size):
            for k in range(speed):
                index = random.randint(0, num_sets-1)
                while y[i, j, index].asscalar() > 0:
                    index = random.randint(0, num_sets-1)

                y[i, j, index] = k+1
                cache_state[index] = 1

                for l in range(k*interval, k*interval+interval):
                    if cache_state[l] == 1:
                        x[i, j, l] = 1
                        cache_state[l] = 0


            # This add victim's access during Set and Check interval
            for k in range(speed):
                if random.random() < pinterval:
                    index = random.randint(0, num_sets-1)
                    while y[i, j, index].asscalar() > 0:
                        index = random.randint(0, num_sets-1)

                    y[i, j, index] = k+1


    return x, y

# comapre the Prime-Probe and Prime-Check
def GenDataPrimeCheck(num_batches, batch_size, num_sets, speed=1, coverage=1, check=1):

    cache_state = [0]*num_sets
    boundary = num_sets/coverage
    interval = boundary/speed

    x = nd.zeros((num_batches, batch_size, num_sets))
    y = nd.zeros((num_batches, batch_size, num_sets))
    for i in range(num_batches):
        for j in range(batch_size):
            for k in range(speed):
                index = random.randint(0, num_sets-1)
                while y[i, j, index].asscalar() > 0:
                    index = random.randint(0, num_sets-1)

                y[i, j, index] = k+1
                cache_state[index] = 1

                for l in range(k*interval, k*interval+interval):
                    if check == 1:
                        if cache_state[l] == 1:
                            x[i, j, l] = 1
                            cache_state[l] = 0
                    else:
                        x[i, j, l] = 1
                        cache_state[l] = 0

    return x, y


def GenDataPartitionWay(num_batches, batch_size, num_sets, speed=1, coverage=1):

    cache_state = [0]*num_sets
    boundary = num_sets/coverage
    interval = boundary/speed

    x = nd.zeros((num_batches, batch_size, num_sets))
    y = nd.zeros((num_batches, batch_size, num_sets))
    for i in range(num_batches):
        for j in range(batch_size):
            for k in range(speed):
                index = random.randint(0, num_sets-1)
                while y[i, j, index].asscalar() > 0:
                    index = random.randint(0, num_sets-1)

                y[i, j, index] = k+1
                cache_state[index] = 1

                for l in range(k*interval, k*interval+interval):
                    x[i, j, l] = 1

    return x, y

def GenDataPartitionSet(num_batches, batch_size, num_sets, speed=1, coverage=1):

    cache_state = [0]*num_sets
    boundary = num_sets/coverage
    interval = boundary/speed

    x = nd.zeros((num_batches, batch_size, num_sets))
    y = nd.zeros((num_batches, batch_size, num_sets))
    for i in range(num_batches):
        for j in range(batch_size):
            for k in range(speed):
                index = random.randint(0, num_sets/2-1)
                while y[i, j, index].asscalar() > 0:
                    index = random.randint(0, num_sets/2-1)

                y[i, j, index] = k+1
                cache_state[index] = 1

    return x, y

def GenDataRandomEviction(num_batches, batch_size, num_sets, speed=1, coverage=1, eviction_speed=0):

    cache_state = [0]*num_sets
    boundary = num_sets/coverage
    interval = boundary/speed

    x = nd.zeros((num_batches, batch_size, num_sets))
    y = nd.zeros((num_batches, batch_size, num_sets))

    speed_ratio = eviction_speed*1.0/speed

    speed_flag = 0
    for i in range(num_batches):
        for j in range(batch_size):
            for k in range(speed):
                index = random.randint(0, num_sets-1)
                while y[i, j, index].asscalar() > 0:
                    index = random.randint(0, num_sets-1)

                y[i, j, index] = k+1
                cache_state[index] = 1

                # conduct random eviction
                if speed_ratio >= 1:
                    for l in range(int(speed_ratio)):
                        index = random.randint(0, num_sets-1)
                        cache_state[index] = 2

                elif speed_ratio > 0:
                    speed_flag += 1
                    period = int(1/speed_ratio)
                    if speed_flag % period == 0:
                        index = random.randint(0, num_sets-1)
                        cache_state[index] = 2
                      

                for l in range(k*interval, k*interval+interval):
                    if cache_state[l] > 0:
                        x[i, j, l] = 1
                        cache_state[l] = 0

    return x, y
