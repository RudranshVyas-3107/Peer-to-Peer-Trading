#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import statistics
from predicted_usage_pj import PredictedUsagePJ


# In[2]:


class QNetwork(tf.keras.Model):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes):
        super(QNetwork, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(input_dim,))
        self.hidden_layers = []
        for size in hidden_layer_sizes:
            self.hidden_layers.append(tf.keras.layers.Dense(size, activation='relu'))
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='linear')

    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


# In[3]:


def change_output_layer_and_retain_weights(base_model, num_neurons):
    input_shape = base_model.input_shape[1:]  
    
    # Rebuild the base model
    base_model = build_base_model(input_shape)
    base_model.load_weights("base_model_weights.h5")  # Load the saved weights
    
    # Add a new output layer with the desired number of neurons
    output_layer = tf.keras.layers.Dense(num_neurons, activation='softmax')
    base_model.add(output_layer)
    
    return base_model


# In[4]:


# def q_learning_prosumer_agent(Q, policy, rewards, max_episodes, max_iterations, epsilon_decay_rate, j,Pt_Hj, rho_t_b, rho_t_s):
#     for episode in range(max_episodes):
#         epsilon = max(1 - episode / epsilon_decay_rate, 0.01)  
        
#         for iteration in range(max_iterations):
#             t = iteration
#             s_t_PAj = get_current_state_prosumer()  
            
#             if np.random.rand() < epsilon:
#                 a_t = np.random.choice(policy[s_t_PAj])  
#             else:
#                 a_t = np.argmax(Q[s_t_PAj])             
#             r_t_plus_1_PAj = get_reward_prosumer(Pt_Hj, rho_t_b, rho_t_s)  
#             s_t_plus_1_PAj = get_next_state()  
            
#             max_q_value = np.max(Q[s_t_plus_1_PAj])
            
#             Q[s_t_PAj][a_t] += alpha_prosumer[j] * (r_t_plus_1_PAj + gamma_prosumer[j] * max_q_value - Q[s_t_PAj][a_t])           
#             s_t_PAj = s_t_plus_1_PAj
def deep_q_learning_prosumer_agent(Q_network, epsilon_decay_rate, Pt_Hj, rho_t_b, rho_t_s, gamma_prosumer, iteration): 
    epsilon = max(0.01, 1 - iteration / epsilon_decay_rate)
        
    s_t_PAj = get_current_state_prosumer(iteration)
    
    q_values = Q_network.predict(np.array([s_t_PAj]))
            
    if np.random.rand() < epsilon:
        a_t = np.random.randint(Q_network.output_shape[1])
    else:
        a_t = np.argmax(q_values)
            
    r_t_plus_1_PAj = get_reward_prosumer(Pt_Hj, rho_t_b, rho_t_s)
    s_t_plus_1_PAj = get_current_state_prosumer(iteration + 1)
            
    q_values_next = Q_network.predict(np.array([s_t_plus_1_PAj]))
    max_q_value = np.max(q_values_next)
            
    target = q_values
    target[0][a_t] = r_t_plus_1_PAj + gamma_prosumer * max_q_value
            
    Q_network.fit(np.array([s_t_PAj]), target, epochs=1, verbose=0)
    
    return a_t


# In[5]:


# def q_learning_grid_agent(Q, policy, rewards, max_episodes, max_iterations, epsilon_decay_rate,Pt_D,Pt_Gi,Pt_Hj,rho_t_s):
#     for episode in range(max_episodes):## episode here can be referred as hour  
#         epsilon = max(1 - episode / epsilon_decay_rate, 0.01)  
        
#         for iteration in range(max_iterations): # iteration can be treated as a minute
#             t = iteration
#             s_t_GA = get_current_state_grid()  
            
#             if np.random.rand() < epsilon:
#                 a_t = np.random.choice(policy[s_t_GA])  
#             else:
#                 a_t = np.argmax(Q[s_t_GA])  
            
#             r_t_plus_1_GA = get_reward_grid(Pt_D,Pt_Gi,Pt_Hj,rho_t_s)  
#             s_t_plus_1_GA = get_next_state()  
            
#             max_q_value = np.max(Q[s_t_plus_1_GA])
            
#             Q[s_t_GA][a_t] += alpha * (r_t_plus_1_GA + gamma * max_q_value - Q[s_t_GA][a_t])  
            
#             s_t_GA = s_t_plus_1_GA
def deep_q_learning_grid_agent(Q_network, epsilon_decay_rate, Pt_D, Pt_Gi, Pt_Hj, rho_t_s, gamma, iteration):
    epsilon = max(0.01, 1 - iteration / epsilon_decay_rate)
    
    s_t_GA = get_current_state_grid(iteration)
    
    q_values = Q_network.predict(np.array([s_t_GA]))
            
    if np.random.rand() < epsilon:
        a_t = np.random.randint(Q_network.output_shape[1])
    else:
        a_t = np.argmax(q_values)
            
    r_t_plus_1_GA = get_reward_grid(Pt_D, Pt_Gi, Pt_Hj, rho_t_s)
    s_t_plus_1_GA = get_current_state_grid(iteration + 1)
            
    q_values_next = Q_network.predict(np.array([s_t_plus_1_GA]))
    max_q_value = np.max(q_values_next)
            
    target = q_values
    target[0][a_t] = r_t_plus_1_GA + gamma * max_q_value
            
    Q_network.fit(np.array([s_t_GA]), target, epochs=1, verbose=0)
    
    return a_t


# In[6]:


def get_reward_grid(Pt_D, Pt_Gi, Pt_Hj, rho_t_s):
    
    vt_G = Pt_D * rho_t_s  
    grid_cost_generation = sum(Pt_Gi)
    grid_cost_prosumers = sum(Pt_Hj)
    rt_GA = vt_G - (grid_cost_generation + grid_cost_prosumers)
    
    return rt_GA


# In[7]:


def get_reward_prosumer(Pt_Hj, rho_t_b, rho_t_s):
    
    vt_Hj = Pt_Hj * rho_t_b
    prosumer_cost = Pt_Hj * rho_t_s
    rt_PAj = vt_Hj - prosumer_cost
    
    return rt_PAj


# In[8]:


def get_current_state_prosumer(i):
    market_price= np.random.rand(60)
    current_state = [
        get_battery_state(i),  # how much our battery is charged at that moment
        get_pv_generation(i),   # how much power has been generated for the past minute/hour/or any scale
        market_price[i],  # current market price
    ]
    
    return current_state


# In[9]:


def get_current_state_grid(i):
    current_state = [
        get_generation_costs(i),  # generation costs
        get_prosumer_costs(i),   # prosumer costs
        get_grid_demand(i)       # grid demand
    ]
    return current_state


# In[10]:


def best_buy_sell_price(data_mp,a_t):
    
    buy_sell_prices={}
    variance = int(np.var(data_mp))
    cmp=int(avg_curerent_market_price(data))
    j=cmp -variance
    
    for i in range(2*variance):
        buy_sell_price['i']=j    
        j=j+1
        
    a_t_key=str(a_t)
    best_price=buy_sell_price[a_t_key]
    
    return best_price


# In[11]:


def avg_current_market_price(data_mp):
    avg = statistics.mean(data_mp)
    return avg


# In[12]:


def grid_buy_price(i):
    grid_buy_prices = [10.0, 9.5, 10.2, 11.0, 10.8, 11.5, 10.2, 10.9, 11.1,
                      10.0, 9.5, 10.2, 11.0, 10.8, 11.5, 10.2, 10.9, 11.1,
                      10.0, 9.5, 10.2, 11.0, 10.8, 11.5, 10.2, 10.9, 11.1,
                      10.0, 9.5, 10.2, 11.0, 10.8, 11.5, 10.2, 10.9, 11.1,
                      10.0, 9.5, 10.2, 11.0, 10.8, 11.5, 10.2, 10.9, 11.1,
                      10.0, 9.5, 10.2, 11.0, 10.8, 11.5, 10.2, 10.9, 11.1]
    return grid_buy_prices[i]


# In[13]:


def get_battery_state(i):
    battery_states = np.random.rand(60)
    return battery_states[i]


# In[14]:


def grid_sell_price(i):
    grid_sell_prices = [12.0, 12.5, 13.0, 12.8, 13.2, 12.7, 13.3, 12.9, 13.0,
                       12.0, 12.5, 13.0, 12.8, 13.2, 12.7, 13.3, 12.9, 13.0,
                       12.0, 12.5, 13.0, 12.8, 13.2, 12.7, 13.3, 12.9, 13.0,
                       12.0, 12.5, 13.0, 12.8, 13.2, 12.7, 13.3, 12.9, 13.0,
                       12.0, 12.5, 13.0, 12.8, 13.2, 12.7, 13.3, 12.9, 13.0,
                       12.0, 12.5, 13.0, 12.8, 13.2, 12.7, 13.3, 12.9, 13.0]
    return grid_sell_prices[i]


# In[15]:


def grid_power(i): # total power demanded from the grid at a particular time t
    grid_power_demand = [50, 48, 55, 60, 52, 58, 62, 56, 53,
                        50, 48, 55, 60, 52, 58, 62, 56, 53,
                        50, 48, 55, 60, 52, 58, 62, 56, 53,
                        50, 48, 55, 60, 52, 58, 62, 56, 53,
                        50, 48, 55, 60, 52, 58, 62, 56, 53,
                        50, 48, 55, 60, 52, 58, 62, 56, 53]
    return grid_power_demand[i]


# In[16]:


def Data(i): # list of current market prices over a certain period of time decided by us
    market_prices = [10.5, 10.2, 11.0, 11.5, 10.8, 11.2, 11.3, 10.9, 11.1,
                    10.5, 10.2, 11.0, 11.5, 10.8, 11.2, 11.3, 10.9, 11.1,
                    10.5, 10.2, 11.0, 11.5, 10.8, 11.2, 11.3, 10.9, 11.1,
                    10.5, 10.2, 11.0, 11.5, 10.8, 11.2, 11.3, 10.9, 11.1,
                    10.5, 10.2, 11.0, 11.5, 10.8, 11.2, 11.3, 10.9, 11.1,
                    10.5, 10.2, 11.0, 11.5, 10.8, 11.2, 11.3, 10.9, 11.1]
    return market_prices[i], market_prices


# In[17]:


def jth_prosumer_power(i): # excess power generated by a prosumer
    prosumer_power_generation = [20, 22, 18, 25, 23, 19, 26, 21, 24,
                               20, 22, 18, 25, 23, 19, 26, 21, 24,
                               20, 22, 18, 25, 23, 19, 26, 21, 24,
                               20, 22, 18, 25, 23, 19, 26, 21, 24,
                               20, 22, 18, 25, 23, 19, 26, 21, 24,
                               20, 22, 18, 25, 23, 19, 26, 21, 24]
    return prosumer_power_generation[i]


# In[18]:


def get_pv_generation(i):
    pv_generation_values = np.random.uniform(0, 30, 60)
    return pv_generation_values[i]


# In[19]:



timestamps = np.arange(60)  # Assuming 60 samples
battery_states = np.random.rand(60)  # Random battery states for each sample
pv_generation = np.random.rand(60)  # Random PV generation for each sample
market_prices = np.random.rand(60)  # Random market prices for each sample
energy_values = np.random.rand(60)  # Random energy values for each sample

# Stack the features into X_train
X_train = np.column_stack((timestamps, battery_states, pv_generation, market_prices))

# Set up y_train as the energy values
y_train = energy_values


# In[20]:


# def main():
#     max_iteration = 100
#     input_dim = 3  # Input dimension for the Q-network
#     predicted_pj = PredictedUsagePJ()
    
#     #ranges for hyperparameters
#     gamma_prosumer_range = [0.9, 0.95, 0.99]
#     gamma_range = [0.9, 0.95, 0.99]
#     epsilon_decay_rate_range = [0.1, 0.01, 0.001]
#     hidden_layer_sizes_range = [[64, 32], [128, 64], [32, 16]]

#     for gamma_prosumer in gamma_prosumer_range:
#         for gamma in gamma_range:
#             for epsilon_decay_rate in epsilon_decay_rate_range:
#                 for hidden_layer_sizes in hidden_layer_sizes_range:

#                     # Initialize Q-networks
#                     q_network_prosumer = QNetwork(input_dim, initial_output_dim, hidden_layer_sizes)
#                     q_network_grid = QNetwork(input_dim, initial_output_dim, hidden_layer_sizes)

#                     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#                     q_network_prosumer.compile(optimizer=optimizer, loss='mean_squared_error')
#                     q_network_grid.compile(optimizer=optimizer, loss='mean_squared_error')
                    
#                     # Define weight initializers
#                     weights_initializer = ''  
#                     activation_function = 'relu'  

#                     for iteration in range(max_iteration):
                        
#                         data,market_prices = Data(iteration)
#                         variance = int(np.var(data))
#                         output_dim = 2 * variance

#                         # Modify output layer dimensions
# #                         q_network_prosumer.output_layer = tf.keras.layers.Dense(output_dim, activation='linear')
# #                         q_network_grid.output_layer = tf.keras.layers.Dense(output_dim, activation='linear')

#                         Pt_Hj = jth_prosumer_power(iteration)
#                         Pt_D = grid_power(iteration)
#                         rho_t_b = grid_buy_price(iteration)
#                         rho_t_s = grid_sell_price(iteration)
                        
#                         a_t_prosumer = deep_q_learning_prosumer_agent(q_network_prosumer, epsilon_decay_rate,
#                                                                      Pt_Hj, rho_t_b, rho_t_s, gamma_prosumer)
#                         a_t_grid = deep_q_learning_grid_agent(q_network_grid, epsilon_decay_rate,
#                                                              Pt_D, Pt_Gi, Pt_Hj, rho_t_s, gamma)

#                         if(i==1 or i==2 or i==3 or i==4 or i==5):
#                             data_mp=market_prices[i:i+5]
#                         else:
#                             data_mp=market_prices[i-4:i+1]
                            
#                         if predicted_usage_pj > average_usage_pj:
#                             best_buy_price, variance = best_buy_sell_price(data_mp, a_t_prosumer)
#                             print(f"Gamma_prosumer: {gamma_prosumer}, Gamma: {gamma}, Epsilon: {epsilon_decay_rate}, Hidden Layers: {hidden_layer_sizes}, Buy Price: {best_buy_price}")
#                         else:
#                             best_sell_price, variance = best_buy_sell_price(data, a_t_prosumer)
#                             print(f"Gamma_prosumer: {gamma_prosumer}, Gamma: {gamma}, Epsilon: {epsilon_decay_rate}, Hidden Layers: {hidden_layer_sizes}, Sell Price: {best_sell_price}")
max_iteration = len(X_train)
input_dim = 3 
epsilon_decay_rate=0.01
predicted_pj = PredictedUsagePJ()
hidden_layer_sizes = [64, 128, 64] ## just examples

initial_output_dim=2

# Initialize Q-networks
q_network_prosumer = QNetwork(input_dim, initial_output_dim, hidden_layer_sizes)
q_network_grid = QNetwork(input_dim, initial_output_dim, hidden_layer_sizes)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
q_network_prosumer.compile(optimizer=optimizer, loss='mean_squared_error')
q_network_grid.compile(optimizer=optimizer, loss='mean_squared_error')

# print(q_network_prosumer.output_shape[1])


for iteration in range(max_iteration):
    data, market_prices = Data(iteration)
    Pt_Hj = jth_prosumer_power(iteration)
    Pt_D = grid_power(iteration)
    rho_t_b = grid_buy_price(iteration)
    rho_t_s = grid_sell_price(iteration)
    gamma_prosumer=0.3  
    
    a_t_prosumer = deep_q_learning_prosumer_agent(q_network_prosumer, epsilon_decay_rate, Pt_Hj, rho_t_b, rho_t_s, gamma_prosumer,iteration)
    a_t_grid = deep_q_learning_grid_agent(q_network_grid, epsilon_decay_rate, Pt_D, Pt_Gi, Pt_Hj, rho_t_s, gamma,iteration)
    
    if iteration >= 5:
        data_mp = market_prices[iteration - 4:iteration + 1]
    else:
        data_mp=market_prices[i:i+5]
        
    x = X_train[iteration]
    y = y_train[iteration]
    
    predicted_pj.train_model(x, y)
    
    buy_sell_price,variance = best_buy_sell_price(data_mp, a_t_prosumer)
    num_output_neurons=2*variance
    
    print("first itr")
    
    # Retain weights and change output layer
    q_network_prosumer = change_output_layer_and_retain_weights(q_network_prosumer, num_output_neurons)
    q_network_grid = change_output_layer_and_retain_weights(q_network_grid, num_output_neurons)
    
    predicted_usage_pj = PredictedUsagePJ.predict_usage(get_current_state_prosumer(iteration + 1))

    if predicted_usage_pj > average_usage_pj:
        best_buy_price, variance = best_buy_sell_price(data_mp, a_t_prosumer)
        print(f"Iteration: {iteration}, Buy Price: {best_buy_price}")
    else:
        best_sell_price, variance = best_buy_sell_price(data_mp, a_t_prosumer)
        print(f"Iteration: {iteration}, Sell Price: {best_sell_price}")
        
    

    


# In[ ]:




