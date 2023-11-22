## Install the following if on a new instance, otherwise they'll ship with the container.
# !pip install nes-py==0.2.6
# !pip install gym-super-mario-bros
# !apt-get update
# !apt-get install ffmpeg libsm6 libxext6  -y

import torch
from tqdm import tqdm
from abc import abstractmethod

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
from ResNet50.ResNet import ResNet
from UNet_Multiclass.scripts.model import UNET as UNet
from torchvision import transforms
from RL.mario_env import *
import collections
from PIL import Image
import numpy as np

from DQN import *

class Preprocessing:
    """
    Define the usege of SS models as pre processing
        - SS    -> Res-net model
        - Unet  -> Unet model
        - shad  -> Shadow frame combination
        - None  -> No pre processing

    Define preprocessing parameters:
        - preprocessing_type
        - model_path
        - in_channels
        - out_channels
    """

    def __init__(self, preprocessing_type, model_path, in_channels, out_channels, device='cpu'):
        self.preprocessing_type = preprocessing_type
        self.model_path = model_path
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.state_img_stack = []

        self._load_model()

    def _load_model(self):
        """
        Load the model from the model_path
        """
        if self.preprocessing_type == 'SS' or self.preprocessing_type == 'shad':
            self.model = ResNet(self.model_path, device=self.device)

        elif self.preprocessing_type == 'Unet':
            self.model = UNet(self.in_channels, self.out_channels)

        else:
            self.model = None

    def process(self, frame):
        img = None

        if self.preprocessing_type == 'SS':
            img = self.model.segment_labels(frame)
            img = np.uint8(img * 255 / 6)

        elif self.preprocessing_type == 'shad':
            img = self.model.segment_labels(frame)
            self.state_img_stack.append(img)
            self.state_img_stack = self.state_img_stack[-4:]
            img = self._merge_frames(self.state_img_stack)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        elif self.preprocessing_type == 'Unet':
            #cv2.imshow('img', frame)
            img = self.model.predict(frame)

            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            img = np.uint8(img * 255 / 6)

        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        return img

    def _merge_frames(self, frames):
        imgs = frames.copy()
        imgs.reverse()

        final_image = Image.fromarray(imgs[0])
        final_image.convert("RGBA")
        final_image.putalpha(255)

        length = len(imgs)

        for i, img in enumerate(imgs):
            alpha = int(180 * (length - i) / length)

            next_image = Image.fromarray(img)
            next_image = next_image.convert("RGBA")
            next_image.putalpha(alpha)
            datas = next_image.getdata()
            new_data = []

            for item in datas:
                if item[0] == 0 and item[1] == 0 and item[2] == 0:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)

            next_image.putdata(new_data)

            final_image.paste(next_image, (0, 0), next_image)

        return np.array(final_image)

class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    And applies semantic segmentation if set to. Otherwise uses grayscale normal frames.
    Returns numpy array
    """
    def __init__(self, pp: Preprocessing, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.pp = pp
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs, self.pp)

    @abstractmethod
    def process(frame, pp: Preprocessing):
        global state_img_stack
        if frame.size == 240 * 256 * 3:
            cv2.imshow('frame', frame)
            img = np.reshape(frame, [240, 256, 3]).astype(np.uint8)

            img = pp.process(img)

        else:
            assert False, "Unknown resolution."

        # Re-scale image to fit model.
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_NEAREST)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])

        return x_t.astype(np.uint8)

def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action
    return [0 for _ in range(action)] + [1] + [0 for _ in range(action + 1, action_space)]


def show_state(env, ep=0, info=""):
    cv2.imshow("Output!",env.render(mode='rgb_array')[:,:,::-1]) #Display using opencv
    cv2.waitKey(1)

def make_env(env,pp):
    env = MaxAndSkipEnv(env)
    #print(env.observation_space.shape)
    env = ProcessFrame84(env=env,pp=pp)
    #print(env.observation_space.shape)

    env = ImageToPyTorch(env)
    #print(env.observation_space.shape)

    env = BufferWrapper(env, 6)
    #print(env.observation_space.shape)

    env = ScaledFloatFrame(env)
    #print(env.observation_space.shape)

    return JoypadSpace(env, RIGHT_ONLY) #Fixes action sets

def run(training_mode, pretrained,
        training_parameters,
        level,
        device='cpu',
        savepath='./results',
        path_dq1=None,
        path_dq2=None,
        vis=True,
        pp: Preprocessing = None):

    backup_interval = training_parameters['backup_interval']
    epochs = training_parameters['epochs']
    run_name = training_parameters['run_name']

    env = gym_super_mario_bros.make('SuperMarioBros-' + level + '-v0')  # Load level
    env = make_env(env, pp)  # Wraps the environment so that frames are grayscale / segmented
    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     max_memory_size=4000,
                     batch_size= training_parameters['batch_size'],
                     gamma=training_parameters['gamma'],
                     lr=training_parameters['learning_rate'],
                     dropout=training_parameters['dropout'],
                     exploration_max=training_parameters['max_exploration_rate'],
                     exploration_min=training_parameters['min_exploration_rate'],
                     exploration_decay=training_parameters['exploration_decay'],
                     pretrained=pretrained,
                     device=device,
                     savepath=savepath,
                     path_dq1=path_dq1,
                     path_dq2=path_dq2)

    # Reset environment
    env.reset()

    # Store rewards and positions
    total_rewards = []
    ending_positions = []

    max_reward = 0

    # Each iteration is an episode (epoch)
    for ep_num in tqdm(range(epochs)):
        global vis_img_stack, state_img_stack
        vis_img_stack = []
        state_img_stack = []

        # Reset state and convert to tensor
        state = env.reset()
        state = torch.Tensor(np.array([state]))

        # Set episode total reward and steps
        total_reward = 0
        steps = 0
        # Until we reach terminal state
        while True:

            # Visualize or not
            if vis:
                show_state(env, ep_num)

            # What action would the agent perform
            action = agent.act(state)
            # Increase step number
            steps += 1
            # Perform the action and advance to the next state
            state_next, reward, terminal, info = env.step(int(action[0]))
            # Update total reward
            total_reward += reward
            # Change to next state
            state_next = torch.Tensor(np.array([state_next]))
            # Change reward type to tensor (to store in ER)
            reward = torch.tensor(np.array([reward])).unsqueeze(0)


            # Is the new state a terminal state?
            terminal = torch.tensor(np.array([int(terminal)])).unsqueeze(0)

            ### Actions performed while training:
            if training_mode:
                # If the episode is finished:
                if terminal:
                    ######################### Model backup section #############################
                    save = False
                    # Backup interval.
                    if ep_num % backup_interval == 0 and ep_num > 0:
                        save = True
                    # Update max reward
                    if max_reward < total_reward:
                        max_reward = total_reward

                    # Save model backup
                    if save == True:

                        with open(savepath + "bp_ending_position.pkl", "wb") as f:
                            pickle.dump(agent.ending_position, f)
                        with open(savepath + "bp_num_in_queue.pkl", "wb") as f:
                            pickle.dump(agent.num_in_queue, f)
                        with open(savepath + run_name + "_bp_total_rewards.pkl", "wb") as f:
                            pickle.dump(total_rewards, f)
                        with open(savepath + run_name + "_bp_ending_positions.pkl", "wb") as f:
                            pickle.dump(ending_positions, f)

                        torch.save(agent.local_net.state_dict(), savepath + "e" + str(ep_num) + "best_performer_dq1.pt")
                        torch.save(agent.target_net.state_dict(), savepath + "e" + str(ep_num) + "best_performer_dq2.pt")

                ######################### End of Model Backup Section #################################
                # Add state to experience replay "dataset"
                agent.remember(state, action, reward, state_next, terminal)
                # Learn from experience replay.
                agent.experience_replay()

            # Update state to current one
            state = state_next

            if terminal:
                break  # End episode loop

        # Store rewards and positions. Print total reward after episode.
        total_rewards.append(total_reward)
        ending_positions.append(agent.ending_position)
        print("Total reward after episode {} is {}".format(ep_num + 1, total_rewards[-1]))

    if training_mode:
        with open(savepath + "ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open(savepath + "num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open(savepath + run_name + "_total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)
        with open(savepath + run_name + "_ending_positions.pkl", "wb") as f:
            pickle.dump(ending_positions, f)
        torch.save(agent.local_net.state_dict(), savepath + "dq1.pt")
        torch.save(agent.target_net.state_dict(), savepath + "dq2.pt")

        with open(savepath + run_name + "_generalization_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)

    env.close()
    # Plot rewards evolution
    if training_mode:
        plt.title("Episodes trained vs. Average Rewards (per 500 eps)")
        plt.plot(total_rewards)
        plt.show()

if __name__ == '__main__':

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        print('Running on the GPU')
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
        print('Running on the MPS')
    else:
        print('Running on the CPU')

    pp = Preprocessing('Unet', '../UNet_Multiclass/models/e40_b32_unet.pth', 3, 6, device)

    training_parameters = {
        "working_dir": './RL/results/',
        "batch_size": 32,
        "gamma": 0.90,
        "dropout": 0.,
        "learning_rate": 0.00025,
        "epochs": 10,
        "backup_interval": 2,
        "max_exploration_rate": 0.8,
        "min_exploration_rate": 0.2,
        "exploration_decay": 0.99,
        "run_name" : "test",
    }

    run(training_mode=True,
        pretrained=False,
        vis=True,
        training_parameters=training_parameters,
        device=device,
        level='1-1',
        savepath=training_parameters["working_dir"],
        pp=pp)





