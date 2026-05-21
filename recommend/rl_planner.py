import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, deque

os.environ['DJANGO_SETTINGS_MODULE'] = 'oj.settings'
import django
django.setup()

from utils.neo4j_client import neo4j_client
from submission.models import Submission, JudgeStatus
from problem.models import Problem, ProblemTag


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state):
        return self.net(state)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(np.array(next_states), dtype=torch.float),
            torch.tensor(dones, dtype=torch.float),
        )

    def __len__(self):
        return len(self.buffer)


class PlannerEnv:
    def __init__(self):
        self.topics = []
        self.topic2idx = {}
        self.num_topics = 0
        self.prereq_graph = None
        self.topic_difficulty = {}
        self.topic_mastery_history = defaultdict(list)
        self._load_from_neo4j()

    def _load_from_neo4j(self):
        client = neo4j_client

        topics_data = client.run_query("MATCH (t:Topic) RETURN t.name AS name ORDER BY name")
        self.topics = [r['name'] for r in topics_data]
        self.num_topics = len(self.topics)
        self.topic2idx = {name: i for i, name in enumerate(self.topics)}
        print(f"RL 环境: {self.num_topics} 个知识点")

        self.prereq_graph = np.zeros((self.num_topics, self.num_topics))
        edges = client.run_query("""
            MATCH (t1:Topic)-[:PREREQUISITE_OF]->(t2:Topic)
            RETURN t1.name AS source, t2.name AS target
        """)
        for r in edges:
            i = self.topic2idx.get(r['source'])
            j = self.topic2idx.get(r['target'])
            if i is not None and j is not None:
                self.prereq_graph[i, j] = 1.0

        diff_data = client.run_query("""
            MATCH (t:Topic)
            WHERE t.calculated_difficulty IS NOT NULL
            RETURN t.name AS name, t.calculated_difficulty AS difficulty
        """)
        diff_map = {'Easy': 0.3, 'Medium': 0.6, 'Hard': 0.9}
        for r in diff_data:
            self.topic_difficulty[r['name']] = diff_map.get(r['difficulty'], 0.5)

    def get_mastery_vector(self, username):
        vector = np.zeros(self.num_topics)

        query = """
        MATCH (u:User {username: $username})-[r:MASTERS]->(t:Topic)
        RETURN t.name AS topic, r.mastery_rate AS rate
        """
        try:
            results = neo4j_client.run_query(query, {'username': username})
            for r in results:
                idx = self.topic2idx.get(r['topic'])
                if idx is not None:
                    vector[idx] = float(r['rate']) if r['rate'] else 0.0
        except Exception:
            pass

        return vector

    def get_topic_from_neighbors(self, current_topic_idx):
        neighbors = np.where(self.prereq_graph[current_topic_idx] > 0)[0]
        if len(neighbors) == 0:
            neighbors = np.where(self.prereq_graph[:, current_topic_idx] > 0)[0]
        if len(neighbors) == 0:
            all_indices = np.arange(self.num_topics)
            neighbors = all_indices[all_indices != current_topic_idx]
        return neighbors

    def simulate_step(self, username, current_mastery, action_topic_idx):
        prereqs = np.where(self.prereq_graph[:, action_topic_idx] > 0)[0]
        avg_prereq = 0.5
        if len(prereqs) > 0:
            avg_prereq = np.mean(current_mastery[prereqs]) if len(prereqs) > 0 else 0.5

        difficulty = self.topic_difficulty.get(
            self.topics[action_topic_idx], 0.5
        )

        current_mastery_on_topic = current_mastery[action_topic_idx]

        success_prob = 0.3 * avg_prereq + 0.3 * current_mastery_on_topic + 0.4 * (1 - difficulty)
        success_prob = np.clip(success_prob, 0.05, 0.95)

        success = np.random.random() < success_prob

        improvement = 0.0
        if success:
            improvement = 0.3 * (1.0 - current_mastery_on_topic)
            improvement += 0.1 * (1.0 - difficulty)
        else:
            if np.random.random() < 0.3:
                improvement = 0.05

        reward = success_prob * 2.0 - 0.5

        next_mastery = current_mastery.copy()
        next_mastery[action_topic_idx] = min(1.0, current_mastery_on_topic + improvement)

        return next_mastery, reward, success

    def build_state(self, mastery_vector, target_topic_idx, step_count, max_steps):
        state_features = []

        state_features.extend(mastery_vector.tolist())

        target_onehot = np.zeros(self.num_topics)
        target_onehot[target_topic_idx] = 1.0
        state_features.extend(target_onehot.tolist())

        progress = np.mean(mastery_vector)
        state_features.append(progress)

        state_features.append(target_onehot.dot(mastery_vector))

        state_features.append(step_count / max(max_steps, 1))

        prereqs = np.where(self.prereq_graph[:, target_topic_idx] > 0)[0]
        avg_prereq_for_target = np.mean(mastery_vector[prereqs]) if len(prereqs) > 0 else 0
        state_features.append(avg_prereq_for_target)

        return np.array(state_features, dtype=np.float32)

    @property
    def state_dim(self):
        return self.num_topics + self.num_topics + 4

    @property
    def action_dim(self):
        return self.num_topics


class DQNTrainer:
    def __init__(self, save_dir='recommend_models'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train(self, episodes=2000, max_steps=10, batch_size=64, lr=0.001):
        env = PlannerEnv()

        state_dim = env.state_dim
        action_dim = env.action_dim

        policy_net = DQN(state_dim, action_dim)
        target_net = DQN(state_dim, action_dim)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        replay_buffer = ReplayBuffer(10000)

        users = neo4j_client.run_query("""
            MATCH (u:User)-[:SUBMITTED]->()-[:FOR]->(p:Problem)
            WITH u, count(p) AS cnt
            WHERE cnt >= 5
            RETURN u.username AS username
            LIMIT 500
        """)
        username_list = [r['username'] for r in users]

        if not username_list:
            username_list = ['default_user']
            print("警告: 没有足够活跃用户，使用默认用户")

        epsilon = 1.0
        epsilon_min = 0.05
        epsilon_decay = 0.997
        gamma = 0.95
        update_target_every = 20
        total_steps = 0

        for episode in range(episodes):
            username = random.choice(username_list)
            mastery = env.get_mastery_vector(username)

            if np.sum(mastery) < 0.01:
                mastery = np.random.random(action_dim) * 0.3

            weakest = np.argmin(mastery)

            valid_targets = []
            for t in range(action_dim):
                if mastery[t] < 0.95:
                    valid_targets.append(t)
            if not valid_targets:
                valid_targets = list(range(action_dim))
            target = random.choice(valid_targets)

            state = env.build_state(mastery, target, 0, max_steps)
            episode_reward = 0

            for step in range(max_steps):
                total_steps += 1

                if np.random.random() < epsilon:
                    action = random.randrange(action_dim)
                else:
                    with torch.no_grad():
                        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                        q_values = policy_net(state_t).squeeze(0).numpy()
                        action = int(np.argmax(q_values))

                next_mastery, reward, success = env.simulate_step(
                    username, mastery, action
                )

                next_state = env.build_state(next_mastery, target, step + 1, max_steps)
                done = (step == max_steps - 1) or (
                    next_mastery[target] > 0.9
                )

                replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                mastery = next_mastery
                episode_reward += reward

                if len(replay_buffer) >= batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        next_q = target_net(next_states).max(1)[0]
                        target_q = rewards + gamma * next_q * (1 - dones)

                    loss = F.smooth_l1_loss(q_values, target_q)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                    optimizer.step()

                if done:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if total_steps % update_target_every == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if (episode + 1) % 200 == 0:
                avg_reward = episode_reward
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.3f}, "
                      f"Epsilon: {epsilon:.3f}, Buffer: {len(replay_buffer)}")

        torch.save(policy_net.state_dict(), os.path.join(self.save_dir, 'dqn_planner.pt'))

        env_data = {
            'topics': env.topics,
            'topic2idx': env.topic2idx,
            'num_topics': env.num_topics,
            'prereq_graph': env.prereq_graph,
            'state_dim': state_dim,
            'action_dim': action_dim,
        }
        with open(os.path.join(self.save_dir, 'rl_env.pkl'), 'wb') as f:
            pickle.dump(env_data, f)

        print("DQN 路径规划模型已保存")
        return policy_net, env_data


def load_dqn_planner(model_dir='recommend_models'):
    model_path = os.path.join(model_dir, 'dqn_planner.pt')
    env_path = os.path.join(model_dir, 'rl_env.pkl')

    if not os.path.exists(model_path) or not os.path.exists(env_path):
        return None, None

    with open(env_path, 'rb') as f:
        env_data = pickle.load(f)

    policy_net = DQN(env_data['state_dim'], env_data['action_dim'])
    policy_net.load_state_dict(torch.load(model_path, map_location='cpu'))
    policy_net.eval()

    return policy_net, env_data


def rl_plan_path(username, target_topic, policy_net, env_data, max_steps=10):
    env = PlannerEnv()
    env.topics = env_data['topics']
    env.topic2idx = env_data['topic2idx']
    env.num_topics = env_data['num_topics']
    env.prereq_graph = env_data['prereq_graph']

    if target_topic not in env.topic2idx:
        return None

    target_idx = env.topic2idx[target_topic]

    mastery = env.get_mastery_vector(username)
    if np.sum(mastery) < 0.01:
        mastery = np.random.random(env.num_topics) * 0.2

    path = []
    current_mastery = mastery.copy()
    state = env.build_state(current_mastery, target_idx, 0, max_steps)
    visited = set()

    for step in range(max_steps):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_values = policy_net(state_t).squeeze(0).numpy()

            for v in visited:
                q_values[v] = -float('inf')
            q_values[target_idx] += 1.0

            action = int(np.argmax(q_values))

        if action in visited:
            remaining = [(i, q_values[i]) for i in range(len(q_values))
                         if i not in visited and q_values[i] > -float('inf')]
            if remaining:
                action = max(remaining, key=lambda x: x[1])[0]
            else:
                break

        visited.add(action)
        topic_name = env.topics[action]
        path.append({
            'topic': topic_name,
            'mastery_before': round(float(current_mastery[action]), 3),
        })

        next_mastery, _, _ = env.simulate_step(username, current_mastery, action)

        current_mastery = next_mastery
        state = env.build_state(current_mastery, target_idx, step + 1, max_steps)

        if current_mastery[target_idx] > 0.85:
            break

    return path


if __name__ == '__main__':
    trainer = DQNTrainer()
    trainer.train(episodes=1000)
