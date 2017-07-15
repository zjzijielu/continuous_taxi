from stochastic_taxi_env_v2 import StochasticTaxiEnv2
env = StochasticTaxiEnv2()

print(env.__doc__)
env.render()

# Some basic imports and setup
import numpy as np, numpy.random as nr, gym
from matplotlib import pyplot as plt
import random
import tensorflow as tf

np.set_printoptions(precision=3)
def begin_grading(): print("\x1b[43m")
def end_grading(): print("\x1b[0m")

# Seed RNGs so you get the same printouts as me
env.seed(0); from gym.spaces import prng; prng.seed(10)
# Generate the episode
env.reset()

def decode(i):
    out = []
    out.append(i % 7)
    i = i // 7
    out.append(i % 7)
    i = i // 7
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    out.reverse()
    return out

reward = 0

#timetable

t_table = {x:x // 4 + 7 + 0.25 * (x % 4) for x in range(48)}
print(t_table)


for t in range(100):
    print("overall reward:", reward)
    env.render()
    a = env.action_space.sample()
    #print(a)
    ob, rew, done, _ = env.step(a)
    print(decode(ob))
    reward += rew

    if done:
        break
assert done
env.render();


# Q-Table Learning

def q_table(num_episodes):

    #Initialize table with all zeros
    Q = np.zeros([env.observation_space.n,env.action_space.n])

    # Set learning parameters
    lr = .6
    y = .95
    e = 0.1

    #create lists to contain total rewards and steps per episode
    #jList = []
    rList = []

    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        t = 0
        #The Q-Table learning algorithm
        if i == num_episodes-1:
            print("##### After Training #####")
        while j < 200:
            if i == num_episodes-1:
                #print("time:", t)
                print("overall reward:", rAll)
                print("time: ", t_table[t])
                env.render()
            j+=1
            #Choose an action by greedily (with noise) picking from Q table
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s,:])
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a)
            #Update Q-Table with new knowledge
            Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
            rAll += r
            s = s1
            if i == num_episodes-1:
                if a==4 or a==5 or a==6 or a==7:
                    t = t
                else:
                    t += 1
            if d == True:
                break
        #jList.append(j)
        if i > 0.95 * num_episodes:
            e = 1. / ((i / 500) + 10)
        rList.append(rAll)

    print("Score over time: " +  str(sum(rList)/num_episodes))
    print("Optimal reward: " + str(rList[-1]))
    print("The best reward ever: ", max(rList))

    x = [i for i in range(num_episodes)]

    plt.plot(x, rList)
    plt.show()

def q_network(num_episodes):

    inputs1 = tf.placeholder(shape=[1, 60025],dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([60025, 9],0,0.01))
    Qout = tf.matmul(inputs1,W)
    predict = tf.argmax(Qout,1)

    nextQ = tf.placeholder(shape=[1,9],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    init = tf.initialize_all_variables()

    # Set learning parameters
    y = .6
    e = 0.1
    # create lists to contain total rewards and steps per episode

    rList = []

    print("start training")
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            # Reset environment and get first new observation
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            # The Q-Network
            while j < 200:
                j += 1
                # Choose an action by greedily (with e chance of random action) from the Q-network
                a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(60025)[s:s + 1]})
                #print(a)
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()

                # Get new state and reward from environment
                s1, r, d, _ = env.step(a[0])
                # Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(60025)[s1:s1 + 1]})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + y * maxQ1
                # Train our network using target and predicted Q values
                _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(60025)[s:s + 1], nextQ: targetQ})
                rAll += r
                s = s1
                if d == True:
                    break
            # reduce epsilon factor
            e = 1. / ((i / 500) + 10)
            rList.append(rAll)
    print("end training")
    x = [i for i in range(num_episodes)]

    plt.plot(x, rList)
    plt.show()
    print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")
    print("Final reward: " + str(rList[-1]))
    print("The best reward ever: ", max(rList))



training_method = {
    'q-table': q_table(200000)
}

train = training_method['q-table']




