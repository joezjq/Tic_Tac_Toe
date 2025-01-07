import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, O0XX_index, Epsilon=0.3, LearningRate=0.1):
        """
        初始化Agent
        :param O0XX_index: 玩家标识 (1 或 2)
        :param Epsilon: 探索率 (epsilon-greedy 策略中的 epsilon)
        :param LearningRate: 学习率 (alpha)
        """
        self.value = np.zeros((3, 3, 3, 3, 3, 3, 3, 3, 3))  # 初始化状态值表
        self.previousState = np.zeros(9)  # 上一个状态
        self.index = O0XX_index  # 玩家标识（1 或 2）
        self.epsilon = Epsilon  # 探索率
        self.alpha = LearningRate  # 学习率

    def reset(self):
        """
        重置Agent的状态
        """
        self.previousState = np.zeros(9)

    def actionTake(self, global_state):
        """
        玩家根据当前全局棋盘选择动作，并更新棋盘
        :param global_state: 当前全局棋盘（共享的棋盘状态）
        :return: 更新后的全局棋盘
        """
        self.previousState = global_state.copy()  # 保存当前全局棋盘为 previousState
        available = np.where(global_state == 0)[0]  # 找出所有空位
        length = len(available)

        if length == 0:
            return global_state  # 如果没有空位，返回当前棋盘（游戏结束）
        else:
            random = np.random.uniform(0, 1)
            if random < self.epsilon:
                # 探索：随机选择一个空位落子
                choose = np.random.randint(length)
                global_state[available[choose]] = self.index  # 落子（使用玩家自己的标识）
            else:
                # 利用：选择价值最大的动作
                tempValue = np.zeros(length)
                for i in range(length):
                    tempState = global_state.copy()
                    tempState[available[i]] = self.index  # 假设在此空位落子
                    tempValue[i] = self.value[tuple(tempState.astype(int))]  # 获取新状态价值
                choose = np.where(tempValue == np.max(tempValue))[0]
                chooseIndex = np.random.randint(len(choose))  # 如果有多个最大值动作
                global_state[available[choose[chooseIndex]]] = self.index
            return global_state

    def valueUpdate(self, State, reward=0):
        """
        使用时间差分学习法 (TD Learning) 更新状态值函数
        """
        self.value[tuple(self.previousState.astype(int))] += \
            self.alpha * (reward + self.value[tuple(State.astype(int))] - 
                          self.value[tuple(self.previousState.astype(int))])
        self.previousState = State.copy()  # 更新 previousState

    def update_epsilon(self, decay_rate=0.99):
        """
        随着训练的进行，逐渐减少探索率
        """
        self.epsilon *= decay_rate


def check_winner(state):
    """
    检查当前棋盘状态是否有赢家、平局或游戏是否继续进行
    """
    state = state.reshape(3, 3)
    for i in range(3):
        if np.all(state[i, :] == 1) or np.all(state[:, i] == 1):
            return 1
        if np.all(state[i, :] == 2) or np.all(state[:, i] == 2):
            return 2
    if np.all(np.diag(state) == 1) or np.all(np.diag(np.fliplr(state)) == 1):
        return 1
    if np.all(np.diag(state) == 2) or np.all(np.diag(np.fliplr(state)) == 2):
        return 2
    if not np.any(state == 0):  # 棋盘已满且没有赢家
        return 0  # 平局
    return -1  # 游戏继续进行


def train(episodes=10000):
    """
    训练两个强化学习Agent对弈
    """
    player1 = Agent(O0XX_index=1, Epsilon=0.3, LearningRate=0.1)  # 玩家1
    player2 = Agent(O0XX_index=2, Epsilon=0, LearningRate=0)  # 玩家2

    win_rate_player1 = []
    win_rate_player2 = []
    draw_rate = []

    results = []

    for episode in range(episodes):
        global_state = np.zeros(9)  # 初始化棋盘
        player1.reset()
        player2.reset()

        while True:
            # 玩家1落子
            global_state = player1.actionTake(global_state)
            winner = check_winner(global_state)
            if winner != -1:  # 游戏结束
                # 根据胜负设置奖励
                if winner == 1:  # 玩家1胜
                    player1.valueUpdate(global_state, reward=1)
                    player2.valueUpdate(global_state, reward=-1)
                elif winner == 2:  # 玩家2胜
                    player1.valueUpdate(global_state, reward=-1)
                    player2.valueUpdate(global_state, reward=1)
                elif winner == 0:  # 平局
                    player1.valueUpdate(global_state, reward=0)
                    player2.valueUpdate(global_state, reward=0)
                results.append(winner)
                break  # 跳出循环

            # 正常值更新
            player1.valueUpdate(global_state)

            # 玩家2落子
            global_state = player2.actionTake(global_state)
            winner = check_winner(global_state)
            if winner != -1:  # 游戏结束
                if winner == 1:  # 玩家1胜
                    player1.valueUpdate(global_state, reward=1)
                    player2.valueUpdate(global_state, reward=-1)
                elif winner == 2:  # 玩家2胜
                    player1.valueUpdate(global_state, reward=-1)
                    player2.valueUpdate(global_state, reward=1)
                elif winner == 0:  # 平局
                    player1.valueUpdate(global_state, reward=0)
                    player2.valueUpdate(global_state, reward=0)
                results.append(winner)
                break

            # 正常值更新
            player2.valueUpdate(global_state)

        # 每100局统计一次胜率和平局率
        if (episode + 1) % 100 == 0:
            total_games = len(results)
            player1_wins = results.count(1)
            player2_wins = results.count(2)
            draws = results.count(0)

            win_rate_player1.append(player1_wins / total_games)
            win_rate_player2.append(player2_wins / total_games)
            draw_rate.append(draws / total_games)

        # 减少探索率
        player1.update_epsilon()
        player2.update_epsilon()

    # 绘制胜率变化图
    plt.figure(figsize=(12, 6))
    plt.plot(win_rate_player1, label="Player 1 Win Rate", color="blue")
    plt.plot(win_rate_player2, label="Player 2 Win Rate", color="red")
    plt.plot(draw_rate, label="Draw Rate", color="green")
    plt.xlabel("Training Episodes (x100)")
    plt.ylabel("Rate")
    plt.title("Training Progress: Win/Draw Rates")
    plt.legend()
    plt.show()

    # 绘制游戏结果分布饼图
    plt.figure(figsize=(6, 6))
    plt.pie([results.count(1), results.count(2), results.count(0)],
            labels=["Player 1 Wins", "Player 2 Wins", "Draws"],
            autopct='%1.1f%%', startangle=90, colors=["blue", "red", "green"])
    plt.title("Final Game Outcomes Distribution")
    plt.show()


if __name__ == "__main__":
    train(episodes=50000)
    print("训练完成！")
