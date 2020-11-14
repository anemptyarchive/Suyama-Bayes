# ch3.5 線形回帰の例

#%%

# 利用ライブラリ
import numpy as np
import matplotlib.pyplot as plt


# ch3.5.1 モデルの構築

#%%

# xのベクトル作成関数を定義
def x_vector(x_n, M):
    x_mn = np.zeros(shape=(M, len(x_n)))
    for m in range(M):
        x_mn[m, :] = np.power(x_n, m)
    return x_mn

#%%

# 観測モデルのパラメータを指定
M_truth = 4
sigma = 1.0
lmd = 1.0 / sigma**2
w_m = np.random.choice(
    np.arange(-1.0, 1.0, step=0.1), size=M_truth, replace=True
).reshape([M_truth, -1])

#%%

# 作図用のx軸の値
x_line = np.arange(-3.0, 3.0, step=0.01)
y_line = np.dot(w_m.T, x_vector(x_line, M_truth)).flatten()

# ノイズを含まない観測モデルを作図
fig = plt.figure(figsize=(12, 8))
plt.plot(x_line, y_line)
plt.suptitle('Observation Model', fontsize=20)
plt.title('M=' + str(M_truth), loc='left', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

#%%

# データをサンプリング
N = 50
smp_x_n = np.random.choice(
    np.arange(min(x_line), max(x_line), step=0.01), size=N, replace=True
)
y_1n = np.dot(w_m.T, x_vector(smp_x_n, M_truth))
y_1n += np.random.normal(loc=0.0, scale=1 / lmd, size=N) # ノイズ成分

#%%

# 観測データの散布図を作成
fig = plt.figure(figsize=(12, 8))
plt.scatter(smp_x_n, y_1n) # 観測データ
plt.plot(x_line, y_line) # 観測モデル
plt.suptitle('Observation Model', fontsize=20)
plt.title('M=' + str(M_truth) + ', N=' + str(N), loc='left', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

#%%

# 事前分布のパラメータを指定
M = 4
m_m = np.zeros((M, 1))
sigma_mm = np.identity(M)
lambda_mm = np.linalg.inv(sigma_mm**2)

# xのベクトル作成
x_mn = x_vector(smp_x_n, M)
x_mline = x_vector(x_line, M)

#%%

# サンプリング数を指定
smp_size = 5

# 事前分布からwをサンプリング
smp_model_arr = np.empty((smp_size, len(x_line)))
for i in range(smp_size):
    # wをサンプリング
    smp_w_m = np.random.multivariate_normal(
        mean=m_m.flatten(), cov=np.linalg.inv(lambda_mm), size=1
    ).reshape(M, -1)
    
    # 出力値を計算
    tmp_y_line = np.dot(smp_w_m.T, x_mline).flatten()
    smp_model_arr[i] = tmp_y_line.copy()

# 事前分布からサンプリングしたモデルを作図
fig = plt.figure(figsize=(12, 8))
for i in range(smp_size):
    plt.plot(x_line, smp_model_arr[i], label=str(i+1)) # 事前分布からサンプリングしたwによるモデル
plt.plot(x_line, y_line, color='blue', linestyle='dotted', label='model') # 観測モデル
plt.title('Sampling from Piror Distribution', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

#%%

# 事後分布のパラメータを計算
lambda_hat_mm = lmd * np.dot(x_mn, x_mn.T) + lambda_mm
tmp_m_m = lmd * np.dot(x_mn, y_1n.T)
tmp_m_m += np.dot(lambda_mm, m_m)
m_hat_m = np.dot(np.linalg.inv(lambda_hat_mm), tmp_m_m)

#%%

# 事後分布からwをサンプリング
smp_model_arr = np.empty((smp_size, len(x_line)))
for i in range(smp_size):
    # wをサンプリング
    smp_w_m = np.random.multivariate_normal(
        mean=m_hat_m.flatten(), cov=np.linalg.inv(lambda_hat_mm), size=1
    ).reshape(M, -1)
    
    # 出力値を計算
    tmp_y_line = np.dot(smp_w_m.T, x_mline).flatten()
    smp_model_arr[i] = tmp_y_line.copy()

# 事後分布からサンプリングしたモデルを作図
fig = plt.figure(figsize=(12, 8))
for i in range(smp_size):
    plt.plot(x_line, smp_model_arr[i], label=str(i+1)) # 事後分布からサンプリングしたwによるモデル
plt.scatter(smp_x_n, y_1n.flatten()) # 観測データ
plt.title('Sampling from Posterior Distribution', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

#%%

# 予測分布のパラメータを計算
sigma2_star_hat_line = np.repeat(1 / lmd, len(x_line))
for i in range(len(x_line)):
    sigma2_star_hat_line[i] += x_mline[:, i].T.dot(np.linalg.inv(lambda_hat_mm)).dot(x_mline[:, i])
mu_star_hat_line = np.dot(m_hat_m.T, x_mline).flatten()

#%%

# 予測分布を作図
fig = plt.figure(figsize=(12, 8))
plt.plot(x_line, mu_star_hat_line, color='orange', label='predict') # 予測分布の期待値
plt.plot(x_line, mu_star_hat_line + np.sqrt(sigma2_star_hat_line), 
         color='#00A968', linestyle='--', label='$+\sigma$') # +sigma
plt.plot(x_line, mu_star_hat_line - np.sqrt(sigma2_star_hat_line), 
         color='#00A968', linestyle='--', label='$-\sigma$') # -sigma
plt.plot(x_line, y_line, linestyle=':', color='blue', label='model') # 観測モデル
plt.scatter(smp_x_n, y_1n.flatten(), color='chocolate') # 観測データ
plt.suptitle('Predict Distribution', fontsize=20)
plt.title('M=' + str(M) + ', N=' + str(N), loc='left', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(min(min(y_1n.flatten()), min(y_line)), max(max(y_1n.flatten()), max(y_line)))
plt.grid()
plt.legend()
plt.show()


#%%

# gif画像でデータ数の変化による学習への影響を確認

# 利用ライブラリ
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# xのベクトル作成関数を定義
def x_vector(x_n, M):
    x_mn = np.zeros(shape=(M, len(x_n)))
    for m in range(M):
        x_mn[m, :] = np.power(x_n, m)
    return x_mn

#%%

# 観測モデルのパラメータを指定
M_truth = 4
sigma = 1.0
lmd = 1.0 / sigma**2
w_m = np.random.choice(
    np.arange(-1.0, 1.0, step=0.1), size=M_truth, replace=True
).reshape([M_truth, -1])

# 作図用のx軸の値
x_line = np.arange(-3.0, 3.0, step=0.01)
y_line = np.dot(w_m.T, x_vector(x_line, M_truth)).flatten()

# ノイズを含まない観測モデルを作図
fig = plt.figure(figsize=(12, 8))
plt.plot(x_line, y_line)
plt.title('Observation Model', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

#%%

# データ数を指定:(試行回数)
N = 100

# 事前分布のパラメータを指定
M = 10
m_m = np.zeros((M, 1))
sigma_mm = np.identity(M)
lambda_mm = np.linalg.inv(sigma_mm**2)

# 作図用のx軸の値
x_mline = x_vector(x_line, M)


# 推論
smp_x_n = np.array([])
y_1n = np.array([[]])
for n in range(N):
    
    # データをサンプリング
    smp_x = np.random.choice(
        np.arange(min(x_line), max(x_line), step=0.01), size=1, replace=True
    )
    smp_x_n = np.append(smp_x_n, smp_x)
    
    # 出力値を計算
    tmp_y = np.dot(w_m.T, x_vector(smp_x_n[n:n+1], M_truth))
    tmp_y += np.random.normal(loc=0.0, scale=1 / lmd, size=1) # ノイズ成分
    y_1n = np.append(y_1n, tmp_y, axis=1)
    
    # xのベクトル作成
    x_mn = x_vector(smp_x_n[n:n+1], M)
    
    # 事後分布のパラメータを更新
    old_lambda_mm = lambda_mm.copy()
    lambda_mm += lmd * np.dot(x_mn, x_mn.T)
    tmp_m_m = lmd * np.dot(x_mn, y_1n[0:1, n:n+1].T)
    tmp_m_m += np.dot(old_lambda_mm, m_m)
    m_m = np.dot(np.linalg.inv(lambda_mm), tmp_m_m)
    
    # 予測分布のパラメータを計算
    sigma2_star_line = np.repeat(1 / lmd, len(x_line))
    for i in range(len(x_line)):
        sigma2_star_line[i] += x_mline[:, i].T.dot(np.linalg.inv(lambda_mm)).dot(x_mline[:, i])
    mu_star_line = np.dot(m_m.T, x_mline).flatten()
    
    # 推移を記録
    if n == 0: # 初回
        # 2次元配列を作成
        sigma2_star_arr = np.array([sigma2_star_line])
        mu_star_arr = np.array([mu_star_line])
    elif n > 0: # 初回以降
        # 配列に追加
        sigma2_star_arr = np.append(sigma2_star_arr, [sigma2_star_line.copy()], axis=0)
        mu_star_arr = np.append(mu_star_arr, [mu_star_line.copy()], axis=0)

#%%

'''
animation.ArtistAnimation()を利用するver
現状だとタイトル(のデータ数)をフレームごとに変更できない
'''

# グラフを初期化
plt.cla()

# 予測分布を作図
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
img = []
for n in range(N):
    
    # nフレーム目のグラフを描画
    ims_predit, = ax.plot(x_line, mu_star_arr[n], color='orange', label='predict') # 予測分布の期待値
    ims_psigma, = ax.plot(x_line, mu_star_arr[n] + np.sqrt(sigma2_star_arr[n]), 
                          color='#00A968', linestyle='--', label='$+\sigma$') # +sigma
    ims_msigma, = ax.plot(x_line, mu_star_arr[n] - np.sqrt(sigma2_star_arr[n]), 
                          color='#00A968', linestyle='--', label='$-\sigma$') # -sigma
    ims_model, = ax.plot(x_line, y_line, linestyle=':', color='blue', label='model') # 観測モデル
    ims_x = ax.scatter(smp_x_n[0:n+1], y_1n[0, 0:n+1].flatten(), color='chocolate') # 観測データ
    
    # グラフの設定
    ax.set_title('M=' + str(M) + ', N=' + str(n+1), loc='left', fontsize=20)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([min(min(y_1n.flatten()), min(y_line)), max(max(y_1n.flatten()), max(y_line))])
    ax.grid(True)
    
    # グラフを格納
    img.append([ims_predit, ims_psigma, ims_msigma, ims_model, ims_x])
fig.suptitle('Predict Distribution', fontsize=20)

# gif画像を作成
ani = animation.ArtistAnimation(fig, img, interval=100)
#ani.save("ch3_5_predict_overfit_n.gif")


#%%

'''
animation.FuncAnimation()を利用するver
現状ではなぜこれでうまくできるのか解ってない
'''

# グラフを初期化
plt.cla()

# 予測分布を作図
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Predict Distribution', fontsize=20)
ax = fig.add_subplot(1, 1, 1)

# 作図処理を関数として定義
def update(n):
    
    # 前フレームのグラフを初期化
    ax.cla()
    
    # nフレーム目のグラフを描画
    ax.plot(x_line, mu_star_arr[n], color='orange', label='predict') # 予測分布の期待値
    ax.plot(x_line, mu_star_arr[n] + np.sqrt(sigma2_star_arr[n]), 
            color='#00A968', linestyle='--', label='$+\sigma$') # +sigma
    ax.plot(x_line, mu_star_arr[n] - np.sqrt(sigma2_star_arr[n]), 
            color='#00A968', linestyle='--', label='$-\sigma$') # -sigma
    ax.plot(x_line, y_line, linestyle=':', color='blue', label='model') # 観測モデル
    ax.scatter(smp_x_n[0:n+1], y_1n[0, 0:n+1].flatten(), color='chocolate') # 観測データ
    
    # グラフの設定
    ax.set_title('M=' + str(M) + ', N=' + str(n+1), loc='left', fontsize=20)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([min(min(y_1n.flatten()), min(y_line)), max(max(y_1n.flatten()), max(y_line))])
    ax.grid(True)
    ax.legend(loc='upper right')
    

# gif画像を作成
ani = animation.FuncAnimation(fig, update, frames=len(mu_star_arr), interval=100)
ani.save("ch3_5_predict_overfit_n.gif")


#%%

# gif画像でパラメータ数の変化による学習への影響を確認

# 利用ライブラリ
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# xのベクトル作成関数を定義
def x_vector(x_n, M):
    x_mn = np.zeros(shape=(M, len(x_n)))
    for m in range(M):
        x_mn[m, :] = np.power(x_n, m)
    return x_mn

#%%

# 観測モデルのパラメータを指定
M_truth = 4
sigma = 1.0
lmd = 1.0 / sigma**2
w_m = np.random.choice(
    np.arange(-1.0, 1.0, step=0.1), size=M_truth, replace=True
).reshape([M_truth, -1])


# 作図用のx軸の値
x_line = np.arange(-3.0, 3.0, step=0.01)
y_line = np.dot(w_m.T, x_vector(x_line, M_truth)).flatten()

# データをサンプリング
N = 10
smp_x_n = np.random.choice(
    np.arange(min(x_line), max(x_line), step=0.01), size=N, replace=True
)
y_1n = np.dot(w_m.T, x_vector(smp_x_n, M_truth))
y_1n += np.random.normal(loc=0.0, scale=1 / lmd, size=N) # ノイズ成分


# 観測データの散布図を作成
fig = plt.figure(figsize=(12, 8))
plt.scatter(smp_x_n, y_1n) # 観測データ
plt.plot(x_line, y_line) # 観測モデル
plt.suptitle('Observation Model', fontsize=20)
plt.title('M=' + str(M_truth) + ', N=' + str(N), loc='left', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

#%%

# パラメータ数（次元数）を指定：(試行回数)
max_M = 25


# 推論
for m in range(max_M):
    
    # 事前分布のパラメータを設定
    m_m = np.zeros((m, 1))
    sigma_mm = np.identity(m)
    lambda_mm = np.linalg.inv(sigma_mm**2)
    
    # xのベクトル作成
    x_mn = x_vector(smp_x_n, m)
    x_mline = x_vector(x_line, m)
    
    # 事後分布のパラメータを更新
    old_lambda_mm = lambda_mm.copy()
    lambda_mm += lmd * np.dot(x_mn, x_mn.T)
    tmp_m_m = lmd * np.dot(x_mn, y_1n.T)
    tmp_m_m += np.dot(old_lambda_mm, m_m)
    m_m = np.dot(np.linalg.inv(lambda_mm), tmp_m_m)
    
    # 予測分布のパラメータを計算
    sigma2_star_line = np.repeat(1 / lmd, len(x_line))
    for i in range(len(x_line)):
        sigma2_star_line[i] += x_mline[:, i].T.dot(np.linalg.inv(lambda_mm)).dot(x_mline[:, i])
    mu_star_line = np.dot(m_m.T, x_mline).flatten()
    
    # 推移を記録
    if m == 0: # 初回
        # 2次元配列を作成
        sigma2_star_arr = np.array([sigma2_star_line])
        mu_star_arr = np.array([mu_star_line])
    elif m > 0: # 初回以降
        # 配列に追加
        sigma2_star_arr = np.append(sigma2_star_arr, [sigma2_star_line.copy()], axis=0)
        mu_star_arr = np.append(mu_star_arr, [mu_star_line.copy()], axis=0)

#%%

'''
animation.FuncAnimation()を利用するver
現状ではなぜこれでうまくできるのか解ってない
'''

# グラフを初期化
plt.cla()

# 予測分布を作図
fig = plt.figure(figsize=(12, 8))
fig.suptitle('Predict Distribution', fontsize=20)
ax = fig.add_subplot(1, 1, 1)

# 作図処理を関数として定義
def update(m):
    
    # 前フレームのグラフを初期化
    ax.cla()
    
    # nフレーム目のグラフを描画
    ax.plot(x_line, mu_star_arr[m], color='orange', label='predict') # 予測分布の期待値
    ax.plot(x_line, mu_star_arr[m] + np.sqrt(sigma2_star_arr[m]), 
            color='#00A968', linestyle='--', label='$+\sigma$') # +sigma
    ax.plot(x_line, mu_star_arr[m] - np.sqrt(sigma2_star_arr[m]), 
            color='#00A968', linestyle='--', label='$-\sigma$') # -sigma
    ax.plot(x_line, y_line, linestyle=':', color='blue', label='model') # 観測モデル
    ax.scatter(smp_x_n, y_1n.flatten(), color='chocolate') # 観測データ
    
    # グラフの設定
    ax.set_title('M=' + str(m+1) + ', N=' + str(N), loc='left', fontsize=20)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([min(min(y_1n.flatten()), min(y_line)), max(max(y_1n.flatten()), max(y_line))])
    ax.grid(True)
    ax.legend(loc='upper right')


# gif画像を作成
ani = animation.FuncAnimation(fig, update, frames=len(mu_star_arr), interval=200)
ani.save("ch3_5_predict_overfit_m.gif")


#%%

# クラスとして実装：未完

'''
作図メソッドにx_vector()を組み込むのが面倒だった
'''

# 線形回帰の実装
class LinearRegression:
    def __init__(self, y_1n, x_mn, lmd):
        # 観測データ
        self.y_1n = y_1n # 出力値
        self.x_mn = x_mn # 入力値
        
        # 観測モデルのパラメータ
        self.lmd = lmd # ノイズの精度:(分散の逆数)
    
    # 事後分布のパラメータ更新メソッド
    def update_posterior_params(self, m_m, sigma_mm):
        # 事前分布
        self.M = len(m_m) # 次元数:(モデルのパラメータ数)
        self.m_m = m_m # 平均パラメータ
        self.lambda_mm = np.linalg.inv(sigma_mm**2) # 精度行列パラメータ
        
        # 事後分布のパラメータを計算
        old_lambda_mm = self.lambda_mm.copy()
        self.lambda_mm += lmd * np.dot(self.x_mn, self.x_mn.T)
        tmp_m_m = self.lmd * np.dot(self.x_mn, self.y_1n.T)
        tmp_m_m += np.dot(old_lambda_mm, self.m_m)
        self.m_m = np.dot(np.linalg.inv(self.lambda_mm), tmp_m_m)
        
    # 予測分布のパラメータ更新メソッド
    def update_predict_params(self, x_star_mn):
        # 未知のデータ
        self.N = x_star_mn.shape[1] # データ数
        self.x_star_mn = x_star_mn # 入力値
        
        # 予測分布のパラメータを計算
        self.sigma2_star_n = np.repeat(1 / self.lmd, self.N)
        for n in range(N):
            self.sigma2_star_n[n] += self.x_star_mn[:, n].T.dot(np.linalg.inv(self.lambda_mm)).dot(self.x_star_mn[:, n])
        self.mu_star_n = np.dot(self.m_m.T, self.x_star_mn).flatten()
    
    # 予測分布の作図メソッド
    def plot_predict_distribution(self, x_line=None, y_line=None):
        # 予測分布を作図
        fig = plt.figure(figsize=(12, 8))
        plt.plot(self.x_star_mn[1], self.mu_star_n, color='orange', label='predict') # 予測分布の期待値
        plt.plot(self.x_star_mn[1], self.mu_star_n + np.sqrt(self.sigma2_star_n), 
                 color='#00A968', linestyle='--', label='$+\sigma$') # 予測分布の期待値+sigma
        plt.plot(self.x_star_mn[1], self.mu_star_n - np.sqrt(self.sigma2_star_n), 
                 color='#00A968', linestyle='--', label='$-\sigma$') # 予測分布の期待値-sigma
        #if x_line != None:これじゃムリ
        #    plt.plot(x_line, y_line, linestyle=':', color='blue', label='model') # 観測モデル
        #plt.scatter(x_n, self.y_1n, color='chocolate') # 観測データ
        plt.suptitle('Predict Distribution', fontsize=20)
        plt.title('M=' + str(self.M) + ', N=' + str(self.N), loc='left')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.legend()
        plt.show()

#%%

# 事前分布のパラメータを指定
N = 50
M = 4
m_m = np.zeros((M, 1))
sigma_mm = np.identity(M)

# xのベクトル作成
x_mn = x_vector(smp_x_n, M)
x_mline = x_vector(x_line, M)

# 推論
model = LinearRegression(y_1n, x_mn, lmd)
model.update_posterior_params(m_m, sigma_mm)
model.update_predict_params(x_mline)
model.plot_predict_distribution(x_line, y_line)


#%%

print('end')

