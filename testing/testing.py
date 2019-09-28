def testing():
    # Turn off exploration while testing
    policy_over_options.epsilon = 0
    for option in range(noptions):
        option_policies[option].temperature = 1e-10

    env = FourRooms()

    nepisodes = 10

    rng = np.random.RandomState(1234)

    reward_list = []

    for episode in range(nepisodes):

        state = env.reset()

        option = policy_over_options.sample(state)
        reward_sum =  0
        for step in range(nsteps):
            
            action = option_policies[option].sample(state)
            
            state, reward, done, _ = env.step(action)
            reward_sum+=reward
            # Termination might occur upon entering new state
            if option_terminations[option].sample(state):
                option = policy_over_options.sample(state)

            if done:
                break
        
        reward_list.append(reward_sum/step)

        print("Goal reached!")
        sleep(2)

    mean =  mean(reward_list)
    reward_error_bar = np.std(reward_list)
    return episode*step, mean, reward_error_bar
        

def plot_initial():
    clear_output(True)
    plt.figure(figsize=(20,6))
    plt.subplot(121)
    plt.title('Evaluation Reward For Options')
    plt.xlabel('Number of Options')
    plt.ylabel('Evaluation Reward')
    poly = np.polyfit(noptions_list,evaluation_reward_list,5)
    poly_y = np.poly1d(poly)(noptions_list)

    plt.plot(noptions_list,poly_y)
    plt.errorbar(noptions_list, poly_y, reward_error_bar, linestyle='None', marker='^')

    plt.grid(True)

    plt.subplot(122)
    plt.title('Total Steps for Options')
    plt.xlabel('Number of Options')
    plt.ylabel('Steps')
    poly = np.polyfit(noptions_list,steps_list,5)
    poly_y = np.poly1d(poly)(noptions_list)
    plt.plot(noptions_list,poly_y)
    plt.grid(True)
    plt.savefig('smooth_evaluation_reward_steps.png')

    # Number of options
    evaluation_reward_list = []
    noptions_list = []
    steps_list = []
    reward_error_bar = []

    for noptions in range(1,31):
        print("noptions", noptions)
        steps, evaluation_reward, error = train_four_rooms(noptions)
        steps_list.append(steps)
        noptions_list.append(noptions)
        evaluation_reward_list.append(evaluation_reward)
        reward_error_bar.append(error)