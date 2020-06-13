import torch
import numba

n_arms = 3
mu_arms = torch.tensor([0.1, 0.2, 0.3])

def method_fully_random_selection(n_epochs=1000, disp_flag=False):
    n_arms = len(mu_arms)
    Q = torch.zeros(n_arms)
    N = torch.zeros(n_arms)
    reward_list = torch.zeros(n_epochs)
    for e in range(n_epochs):
        arm_t = torch.randint(n_arms, (1,))
        arm = arm_t[0]
        reward = torch.bernoulli(mu_arms[arm])
        reward_list[e] = reward
        N[arm] += 1
        Q[arm] = (Q[arm]*(N[arm]-1) + reward)/N[arm]
        if disp_flag:
            print(arm_t[0], reward, ' --> ', Q)
    return Q, N, reward_list


def method_greedy_selection_one(n_epochs=1000, disp_flag=False):
    n_arms = len(mu_arms)
    Q = torch.zeros(n_arms)
    N = torch.zeros(n_arms)
    reward_list = torch.zeros(n_epochs)
    arm_t = torch.randint(n_arms, (1,))
    arm = arm_t[0]
    for e in range(n_epochs):
        reward = torch.bernoulli(mu_arms[arm])
        reward_list[e] = reward
        N[arm] += 1
        Q[arm] = (Q[arm]*(N[arm]-1) + reward)/N[arm]
        arm = torch.argmax(Q)
        if disp_flag:
            print(arm, reward, ' --> ', Q)
    return Q, N,reward_list

def method_greedy_selection(n_epochs=1000, n_trials=2, disp_flag=False):
    n_arms = len(mu_arms)
    Q_ntrials = torch.zeros(n_arms, n_trials)
    N_ntrials = torch.zeros(n_arms, n_trials)
    reward_list_ntrials = torch.zeros(n_epochs, n_trials)
    for t in range(n_trials):
        Q, N, reward_list = method_greedy_selection_one(n_epochs, disp_flag)
        Q_ntrials[:,t] = Q
        N_ntrials[:,t] = N
        reward_list_ntrials[:,t] = reward_list
    Q = torch.mean(Q_ntrials,1)
    N = torch.mean(N_ntrials,1)
    reward_list = torch.mean(reward_list_ntrials,1)
    return Q, N,reward_list



def main():
    n_epochs = 1000
    n_trials = 10

    print('------------------------------------')
    print('Method: fully random selection')
    Q, N, reward_list = method_fully_random_selection(n_epochs)
    print()
    print(f'Numerical results with {n_epochs} epochs')
    print(f'Q={Q}')
    print(f'E[N]={N/n_epochs}')
    print(f'total gain={torch.sum(reward_list)}')
    print()
    print(f'Theorecical values with infinity epochs')
    print(f'Q={mu_arms}')
    print(f'E[N]=', [1/n_arms]*n_arms)
    print(f'total gain={torch.mean(mu_arms)*n_epochs}')
    print()

    print('------------------------------------')
    print('Method: greedy selection')
    Q, N, reward_list = method_greedy_selection(n_epochs,n_trials)
    print()
    print(f'Numerical results with {n_epochs} epochs, {n_trials} traials')
    print(f'Q={Q}')
    print(f'E[N]={N/n_epochs}')
    print(f'total gain={torch.sum(reward_list)}')

if __name__ == '__main__':
    main()

