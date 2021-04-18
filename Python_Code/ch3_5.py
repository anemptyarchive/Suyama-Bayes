# ch3.5 線形回帰の例

#%%

# 3.5節で利用ライブラリ
import numpy as np
import matplotlib.pyplot as plt

#%%

## ch3.5.1 モデルの構築

# {x^(m-1)}ベクトル作成関数を定義
def x_vector(x_n, M):
    # 受け皿を作成
    x_nm = np.zeros(shape=(len(x_n), M))
    
    # m乗を計算
    for m in range(M):
        x_nm[:, m] = np.power(x_n, m)
    return x_nm

#%%

# 真の次元数を指定
M_truth = 4

# 真のパラメータを生成
w_truth_m = np.random.choice(
    np.arange(-1.0, 1.0, step=0.1), size=M_truth, replace=True
)
print(w_truth_m)


# 作図用のx軸の値を作成
x_line = np.arange(-3.0, 3.0, step=0.01)

# 作図用のxをM次元に拡張
x_truth_arr = x_vector(x_line, M_truth)

# 真のモデルの出力を計算
y_line = np.dot(w_truth_m.reshape((1, M_truth)), x_truth_arr.T).flatten()
print(y_line[:5])

#%%

# 真のモデルを作図
fig = plt.figure(figsize=(12, 9))
plt.plot(x_line, y_line) # 真のモデル
plt.suptitle('Observation Model', fontsize=20)
plt.title('w=[' + ', '.join([str(w) for w in np.round(w_truth_m, 2).flatten()]) + ']', loc='left', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid() # グリッド線
plt.show()

#%%

## 観測データの生成

# (観測)データ数を指定
N = 50

# 入力値を生成
x_n = np.random.choice(
    np.arange(min(x_line), max(x_line), step=0.01), size=N, replace=True
)

# 入力値をM次元に拡張
x_truth_nm = x_vector(x_n, M_truth)


# ノイズ成分の標準偏差を指定
sigma = 1.5

# ノイズ成分の精度を計算
lmd = 1.0 / sigma**2
print(lmd)

# ノイズ成分を生成
epsilon_n = np.random.normal(loc=0.0, scale=np.sqrt(1 / lmd), size=N)

# 出力値を計算:式(3.141)
y_n = np.dot(w_truth_m.reshape((1, M_truth)), x_truth_nm.T).flatten() + epsilon_n
print(y_n[:5])

#%%

# 観測データの散布図を作成
fig = plt.figure(figsize=(12, 9))
plt.scatter(x_n, y_n) # 観測データ
plt.plot(x_line, y_line) # 真のモデル
plt.suptitle('Observation Data', fontsize=20)
plt.title('$N=' + str(N) + 
          ', w=[' + ', '.join([str(w) for w in np.round(w_truth_m, 2).flatten()]) + ']' + 
          ', \lambda=' + str(np.round(lmd, 2)) + '$', loc='left', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid() # グリッド線
plt.show()

#%%

## 事前分布(多次元ガウス分布)の設定

# 事前分布の次元数を指定
M = 5

# 事前分布の平均を指定
m_m = np.zeros(M)

# 事前分布の精度行列を指定
sigma_mm = np.identity(M) * 10
lambda_mm = np.linalg.inv(sigma_mm**2)
print(lambda_mm)

# 入力値をM次元に拡張
x_nm = x_vector(x_n, M)

# 作図用のxをM次元に拡張
x_arr = x_vector(x_line, M)

#%%

# サンプリング数を指定
smp_size = 5

# 事前分布からサンプリングしたwを用いたモデルを比較
prior_list = []
for i in range(smp_size):
    # パラメータを生成
    prior_w_m = np.random.multivariate_normal(
        mean=m_m, cov=np.linalg.inv(lambda_mm), size=1
    )
    
    # 出力値を計算:式(3.141)
    tmp_y_line = np.dot(prior_w_m.reshape((1, M)), x_arr.T).flatten()
    
    # 結果を格納
    prior_list.append(list(tmp_y_line))


# 事前分布からサンプリングしたパラメータによるモデルを作図
fig = plt.figure(figsize=(12, 9))
for i in range(smp_size):
    plt.plot(x_line, prior_list[i], label='smp:' + str(i+1)) # サンプリングしたwを用いたモデル
plt.plot(x_line, y_line, color='blue', linestyle='dashed', label='model') # 真のモデル
plt.suptitle('Sampling from Prior Distribution', fontsize=20)
plt.title('m=[' + ', '.join([str(m) for m in m_m]) + ']', loc='left', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim((min(y_line) - 3 * sigma, max(y_line) + 3 * sigma)) # y軸の表示範囲
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# 事後分布のパラメータを計算:式(3.148)
lambda_hat_mm = lmd * np.dot(x_nm.T, x_nm) + lambda_mm
term_m_m = lmd * np.dot(y_n.reshape((1, N)), x_nm).T
term_m_m += np.dot(lambda_mm, m_m.reshape((M, 1)))
m_hat_m = np.dot(np.linalg.inv(lambda_hat_mm), term_m_m).flatten()
print(lambda_mm)
print(m_hat_m)

#%%

# 事後分布からサンプリングしたwを用いたモデルを比較
posterior_list = []
for i in range(smp_size):
    # パラメータを生成
    posterior_w_m = np.random.multivariate_normal(
        mean=m_hat_m, cov=np.linalg.inv(lambda_hat_mm), size=1
    )
    
    # 出力値を計算:式(3.141)
    tmp_y_line = np.dot(posterior_w_m.reshape((1, M)), x_arr.T).flatten()
    
    # 結果を格納
    posterior_list.append(list(tmp_y_line))


# 事後分布からサンプリングしたパラメータによるモデルを作図
fig = plt.figure(figsize=(12, 9))
for i in range(smp_size):
    plt.plot(x_line, posterior_list[i], label='smp' + str(i+1)) # サンプリングしたwを用いたモデル
plt.plot(x_line, y_line, color='blue', linestyle='dashed', label='model') # 真のモデル
plt.scatter(x_n, y_n) # 観測データ
plt.suptitle('Sampling from Posterior Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{m}=[' + ', '.join([str(m) for m in np.round(m_hat_m, 2)]) + ']$', 
          loc='left', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim((min(y_line) - 3 * sigma, max(y_line) + 3 * sigma)) # y軸の表示範囲
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# 予測分布のパラメータを計算:式(3.155')
mu_star_hat_line = np.dot(m_hat_m.reshape((1, M)), x_arr.T).flatten()
sigma2_star_hat_line = np.diag(
    x_arr.dot(np.linalg.inv(lambda_hat_mm)).dot(x_arr.T)
) + 1 / lmd
print(mu_star_hat_line[:5])
print(sigma2_star_hat_line[:5])

#%%

# 予測分布を作図
fig = plt.figure(figsize=(12, 9))
plt.plot(x_line, mu_star_hat_line, color='orange', label='predict') # 予測分布の期待値
plt.fill_between(x=x_line, 
                 y1=mu_star_hat_line - np.sqrt(sigma2_star_hat_line), 
                 y2=mu_star_hat_line + np.sqrt(sigma2_star_hat_line), 
                 color='#00A968', alpha=0.3, linestyle='dotted', label='$\mu \pm \sigma$') # 予測分布の標準偏差
plt.plot(x_line, y_line, linestyle='dashed', color='blue', label='model') # 真のモデル
plt.scatter(x_n, y_n, color='chocolate') # 観測データ
plt.suptitle('Predict Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{m}=[' + ', '.join([str(m) for m in np.round(m_hat_m, 2)]) + ']$', 
          loc='left', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim((min(y_line) - 3 * sigma, max(y_line) + 3 * sigma)) # y軸の表示範囲
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()


#%%

### アニメーション

# ・サンプルサイズによる分布の変化をアニメーションで確認

# 利用するライブラリ
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## モデルの設定

# {x^(m-1)}ベクトル作成関数を定義
def x_vector(x_n, M):
    # 受け皿を作成
    x_nm = np.zeros(shape=(len(x_n), M))
    
    # m乗を計算
    for m in range(M):
        x_nm[:, m] = np.power(x_n, m)
    return x_nm


# 真の次元数を指定
M_truth = 4

# ノイズ成分の標準偏差を指定
sigma = 1.5

# ノイズ成分の精度を計算
lmd = 1.0 / sigma**2
print(lmd)

# 真のパラメータを生成
w_truth_m = np.random.choice(
    np.arange(-1.0, 1.0, step=0.1), size=M_truth, replace=True
)
print(w_truth_m)


# 作図用のx軸の値を作成
x_line = np.arange(-3.0, 3.0, step=0.01)

# 真のモデルの出力を計算
y_line = np.dot(w_truth_m.reshape((1, M_truth)), x_vector(x_line, M_truth).T).flatten()

# 真のモデルを作図
fig = plt.figure(figsize=(12, 9))
plt.plot(x_line, y_line) # 真のモデル
plt.suptitle('Observation Model', fontsize=20)
plt.title('w=[' + ', '.join([str(w) for w in np.round(w_truth_m, 2).flatten()]) + ']', loc='left', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid() # グリッド線
plt.show()

#%%

## 事前分布の設定

# 事前分布の次元数を指定
M = 5

# 事前分布のパラメータを指定
m_m = np.zeros(M)
lambda_mm = np.identity(M) * 0.01


# 作図用のxをM次元に拡張
x_arr = x_vector(x_line, M)

# 初期値による予測分布のパラメータを計算:式(3.155)
mu_star_line = np.dot(m_m.reshape((1, M)), x_arr.T).flatten()
sigma2_star_line = np.diag(
    x_arr.dot(np.linalg.inv(lambda_mm)).dot(x_arr.T)
) + 1 / lmd


## 推論処理

# データ数を指定:(試行回数)
N = 100

# 推移の記録用の受け皿を初期化
x_n = np.empty(N)
y_n = np.empty(N)
trace_m = [list(m_m)]
trace_mu_star = [list(mu_star_line)]
trace_sigma_star = [list(np.sqrt(sigma2_star_line))]

# ベイズ推論
for n in range(N):
    
    # 入力値を生成
    x_n[n] = np.random.choice(
        np.arange(min(x_line), max(x_line), step=0.01), size=1, replace=True
    )
    
    # 出力値を計算:式(3.141)
    term_y = np.dot(w_truth_m.reshape((1, M_truth)), x_vector(x_n[n].reshape((1, 1)), M_truth).T)
    term_eps = np.random.normal(loc=0.0, scale=np.sqrt(1 / lmd), size=1) # ノイズ成分
    y_n[n] = term_y + term_eps
    
    # 入力値をM次元に拡張
    x_1n = x_vector(x_n[n].reshape((1, 1)), M)
    
    # 事後分布のパラメータを更新:式(3.148)
    old_lambda_mm = lambda_mm.copy()
    lambda_mm += lmd * np.dot(x_1n.T, x_1n)
    term_m_m = lmd * y_n[n] * x_1n.T
    term_m_m += np.dot(old_lambda_mm, m_m.reshape((M, 1)))
    m_m = np.dot(np.linalg.inv(lambda_mm), term_m_m).flatten()
    
    # 予測分布のパラメータを更新:式(3.155)
    mu_star_line = np.dot(m_m.reshape((1, M)), x_arr.T).flatten()
    sigma2_star_line = np.diag(
        x_arr.dot(np.linalg.inv(lambda_mm)).dot(x_arr.T)
    ) + 1 / lmd
    
    # パラメータを記録
    trace_m.append(list(m_m))
    trace_mu_star.append(list(mu_star_line))
    trace_sigma_star.append(list(np.sqrt(sigma2_star_line)))

#%%

## 予測分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(n):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の予測分布を作図
    plt.plot(x_line, trace_mu_star[n], color='orange', label='predict') # 予測分布の期待値
    plt.fill_between(x=x_line, 
                     y1=np.array(trace_mu_star[n]) - np.array(trace_sigma_star[n]), 
                     y2=np.array(trace_mu_star[n]) + np.array(trace_sigma_star[n]), 
                      color='#00A968', alpha=0.3, linestyle='dotted', label='$\mu \pm \sigma$') # 予測分布の標準偏差
    plt.plot(x_line, y_line, linestyle='dashed', color='blue', label='model') # 真のモデル
    plt.scatter(x_n[0:n], y_n[0:n], color='chocolate') # 観測データ
    plt.xlabel('x')
    plt.ylabel('y')
    plt.suptitle('Predict Distribution', fontsize=20)
    plt.title('$N=' + str(n) + 
              ', \hat{m}=[' + ', '.join([str(m) for m in np.round(trace_m[n], 2)]) + ']$', 
              loc='left', fontsize=20)
    plt.ylim((min(y_line) - 3 * sigma, max(y_line) + 3 * sigma)) # y軸の表示範囲
    plt.grid() # グリッド線

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=N + 1, interval=100)
predict_anime.save("ch3_5_Predict_N.gif")


#%%

# ・次元数による分布の変化をアニメーションで確認

# 利用するライブラリ
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%

## モデルの設定

# {x^(m-1)}ベクトル作成関数を定義
def x_vector(x_n, M):
    # 受け皿を作成
    x_nm = np.zeros(shape=(len(x_n), M))
    
    # m乗を計算
    for m in range(M):
        x_nm[:, m] = np.power(x_n, m)
    return x_nm


# 真の次元数を指定
M_truth = 4

# ノイズ成分の標準偏差を指定
sigma = 1.5

# ノイズ成分の精度を計算
lmd = 1.0 / sigma**2
print(lmd)

# 真のパラメータを生成
w_truth_m = np.random.choice(
    np.arange(-1.0, 1.0, step=0.1), size=M_truth, replace=True
)
print(w_truth_m)

# 作図用のx軸の値を作成
x_line = np.arange(-3.0, 3.0, step=0.01)

# 真のモデルの出力を計算
y_line = np.dot(w_truth_m.reshape((1, M_truth)), x_vector(x_line, M_truth).T).flatten()


## 観測データの生成

# (観測)データ数を指定
N = 10

# 入力値を生成
x_n = np.random.choice(
    np.arange(min(x_line), max(x_line), step=0.01), size=N, replace=True
)

# ノイズ成分を生成
epsilon_n = np.random.normal(loc=0.0, scale=np.sqrt(1 / lmd), size=N)

# 出力値を計算:式(3.141)
y_n = np.dot(w_truth_m.reshape(1, M_truth), x_vector(x_n, M_truth).T) + epsilon_n

# 観測データの散布図を作成
fig = plt.figure(figsize=(12, 9))
plt.scatter(x_n, y_n) # 観測データ
plt.plot(x_line, y_line) # 真のモデル
plt.suptitle('Observation Model', fontsize=20)
plt.title('$N=' + str(N) + 
          ', w=[' + ', '.join([str(w) for w in np.round(w_truth_m, 2).flatten()]) + ']' + 
          ', \lambda=' + str(np.round(lmd, 2)) + '$', loc='left', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.grid() # グリッド線
plt.show()

#%%

## 推論処理

# 事前分布の次元数の最大値(試行回数)を指定
M_max = 15

# 推移の記録用の受け皿を初期化
trace_m = []
trace_mu_star = []
trace_sigma_star = []

# ベイズ推論
for m in range(1, M_max + 1):
    
    # 事前分布のパラメータをm次元に初期化
    m_m = np.zeros(m)
    lambda_mm = np.identity(m) * 0.01
    
    # 入力値をm次元に拡張
    x_nm = x_vector(x_n, m)
    
    # 作図用のｘをm次元に拡張
    x_arr = x_vector(x_line, m)
    
    # 事後分布のパラメータを更新:式(3.148)
    lambda_hat_mm = lmd * np.dot(x_nm.T, x_nm) + lambda_mm
    term_m_m = lmd * np.dot(y_n.reshape((1, N)), x_nm).T
    term_m_m += np.dot(lambda_mm, m_m.reshape((m, 1)))
    m_hat_m = np.dot(np.linalg.inv(lambda_hat_mm), term_m_m).flatten()
    
    # 予測分布のパラメータを計算:式(3.155')
    mu_star_hat_line = np.dot(m_hat_m.reshape((1, m)), x_arr.T).flatten()
    sigma2_star_hat_line = np.diag(
        x_arr.dot(np.linalg.inv(lambda_hat_mm)).dot(x_arr.T)
    ) + 1 / lmd
    
    # パラメータを記録
    trace_m.append(list(m_hat_m))
    trace_mu_star.append(list(mu_star_hat_line))
    trace_sigma_star.append(list(np.sqrt(sigma2_star_hat_line)))

#%%

## 予測分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_predict(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # nフレーム目の予測分布を作図
    plt.plot(x_line, trace_mu_star[i], color='orange', label='predict') # 予測分布の期待値
    plt.fill_between(x=x_line, 
                     y1=np.array(trace_mu_star[i]) - np.array(trace_sigma_star[i]), 
                     y2=np.array(trace_mu_star[i]) + np.array(trace_sigma_star[i]), 
                      color='#00A968', alpha=0.3, linestyle='dotted', label='$\mu \pm \sigma$') # 予測分布の標準偏差
    plt.plot(x_line, y_line, linestyle='dashed', color='blue', label='model') # 真のモデル
    plt.scatter(x_n, y_n, color='chocolate') # 観測データ
    plt.xlabel('x')
    plt.ylabel('y')
    plt.suptitle('Predict Distribution', fontsize=20)
    plt.title('$N=' + str(N) + ', M=' + str(i + 1) + 
              ', \hat{m}=[' + ', '.join([str(m) for m in np.round(trace_m[i], 2)]) + ']$', 
              loc='left', fontsize=20)
    plt.ylim((min(y_line) - 3 * sigma, max(y_line) + 3 * sigma)) # y軸の表示範囲
    plt.grid() # グリッド線

# gif画像を作成
predict_anime = animation.FuncAnimation(fig, update_predict, frames=M_max, interval=100)
predict_anime.save("ch3_5_Predict_M.gif")


#%%

print('end')

