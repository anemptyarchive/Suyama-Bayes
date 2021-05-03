# 4.3.3 ポアソン混合分布における推論：変分推論

#%%

# 4.3.3項で利用するライブラリ
import numpy as np
from scipy.stats import poisson, gamma # ポアソン分布, ガンマ分布
from scipy.special import psi # ディガンマ関数
import matplotlib.pyplot as plt

#%%

## 観測モデル(ポアソン混合分布)の設定

# 真のパラメータを指定
lambda_truth_k  = np.array([10, 25, 40])

# 真の混合比率を指定
pi_truth_k = np.array([0.35, 0.25, 0.4])

# クラスタ数を取得
K = len(lambda_truth_k)


# 作図用のxの点を作成
x_line = np.arange(0, 2 * np.max(lambda_truth_k))
print(x_line)

# 観測モデルを計算
model_prob = 0.0
for k in range(K):
    # クラスタkの分布の確率を計算
    tmp_prob = poisson.pmf(k=x_line, mu=lambda_truth_k[k])
    
    # K個の分布の加重平均を計算
    model_prob += tmp_prob * pi_truth_k[k]

#%%

# 観測モデルを作図
plt.figure(figsize=(12, 9))
plt.bar(x=x_line, height=model_prob) # 真の分布
plt.xlabel('x')
plt.ylabel('prob')
plt.suptitle('Poisson Mixture Model', size = 20)
plt.title('$\lambda=[' + ', '.join([str(lmd) for lmd in lambda_truth_k]) + ']' + 
          ', \pi=[' + ', '.join([str(pi) for pi in pi_truth_k])+ ']$', loc='left')
plt.show()

#%%

## 観測データの生成

# (観測)データ数を指定
N = 250

# 真のクラスタを生成
s_truth_nk = np.random.multinomial(n=1, pvals=pi_truth_k, size=N)

# 真のクラスタ番号を抽出
_, s_truth_n = np.where(s_truth_nk == 1)

# (観測)データを生成
#x_n = np.random.poisson(lam=np.prod(lambda_truth_k**s_truth_nk, axis=1), size=N)
x_n = np.random.poisson(lam=lambda_truth_k[s_truth_n], size=N)
print(x_n[:10])

#%%

# 観測データのヒストグラムを作成
plt.figure(figsize=(12, 9))
plt.bar(x=x_line, height=model_prob, label='true model', 
        color='white', alpha=1, edgecolor='red', linestyle='--') # 真の分布
plt.bar(x=x_line, height=[np.sum(x_n == x) / len(x_n) for x in x_line], label='observation data') # 観測データ
plt.xlabel('x')
plt.ylabel('dens')
plt.suptitle('Poisson Mixture Model', size=20)
plt.title('$N=' + str(N) + 
          ', \lambda=[' + ', '.join([str(lmd) for lmd in lambda_truth_k]) + ']' + 
          ', \pi=[' + ', '.join([str(pi) for pi in pi_truth_k]) + ']$', loc='left')
plt.legend()
plt.show()

#%%

# 真のクラスタのヒストグラムを作成
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.bar(x=x_line, height=[np.sum(x_n[s_truth_n == k] == x) for x in x_line], 
            alpha=0.5, label='cluster:' + str(k + 1)) # 真のクラスタ
plt.xlabel('x')
plt.ylabel('count')
plt.suptitle('Poisson Mixture Model', size=20)
plt.title('$N=' + str(N) + 
          ', \lambda=[' + ', '.join([str(lmd) for lmd in lambda_truth_k]) + ']' + 
          ', \pi=[' + ', '.join([str(pi) for pi in pi_truth_k]) + ']$', loc='left')
plt.legend()
plt.show()

#%%

## 事前分布(ガンマ分布とディリクレ分布)の設定

# lambdaの事前分布のパラメータを指定
a = 1.0
b = 1.0

# piの事前分布のパラメータを指定
alpha_k = np.repeat(2.0, K)

#%%

## 初期値の設定

# 潜在変数の事後分布の期待値の初期値を生成
E_s_nk = np.random.rand(N, K)
E_s_nk /= np.sum(E_s_nk, axis=1, keepdims=True)

# 初期値によるlambdaの事後分布のパラメータを計算:式(4.55)
a_hat_k = np.sum(E_s_nk.T * x_n, axis=1) + a
b_hat_k = np.sum(E_s_nk, axis=0) + b
print(a_hat_k)
print(b_hat_k)

# 初期値によるpiのパラメータを計算:式(4.58)
alpha_hat_k = np.sum(E_s_nk, axis=0) + alpha_k
print(alpha_hat_k)

#%%

# 作図用のlambdaの点を作成
lambda_line = np.linspace(0, 2 * np.max(lambda_truth_k), num=1000)

# 初期値によるlambdaの近似事後分布を作図
posterior_lambda_kl = np.empty((K, len(lambda_line)))
for k in range(K):
    posterior_lambda_kl[k] = gamma.pdf(x=lambda_line, a=a_hat_k[k], scale=1 / b_hat_k[k])

# 初期値によるlambdaの近似事後分布を作図
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.vlines(x=lambda_truth_k[k], ymin=0.0, ymax=np.max(posterior_lambda_kl), 
               color='red', linestyle='--') # 真の値
    plt.plot(lambda_line, posterior_lambda_kl[k], label='cluster:' + str(k + 1)) # 事後分布
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', size=20)
plt.title('$iter:' + str(0) + ', N=' + str(N) + 
          ', \hat{a}=[' + ', '.join([str(a) for a in np.round(a_hat_k, 1)]) + ']' + 
          ', \hat{b}=[' + ', '.join([str(b) for b in np.round(b_hat_k, 1)]) + ']$', loc='left')
plt.legend()
plt.show()

#%%

# lambdaの平均値を計算:式(2.59)
E_lambda_k = a_hat_k / b_hat_k

# piの平均値を計算:式(2.51)
E_pi_k = alpha_hat_k / np.sum(alpha_hat_k)

# 初期値による混合分布を計算
init_prob = 0.0
for k in range(K):
    # クラスタkの分布の確率を計算
    tmp_prob = poisson.pmf(k=x_line, mu=E_lambda_k[k])
    
    # K個の分布の加重平均を計算
    init_prob += tmp_prob * E_pi_k[k]

# 初期値による分布を作図
plt.figure(figsize=(12, 9))
plt.bar(x=x_line, height=model_prob, label='true model', 
        color='white', alpha=1, edgecolor='red', linestyle='--') # 真の分布
plt.bar(x=x_line, height=init_prob, color='purple') # 初期値による分布
plt.xlabel('x')
plt.ylabel('prob')
plt.suptitle('Poisson Mixture Model', size = 20)
plt.title('$iter:' + str(0) + 
          ', E[\lambda]=[' + ', '.join([str(lmd) for lmd in np.round(E_lambda_k, 2)]) + ']' + 
          ', E[\pi]=[' + ', '.join([str(pi) for pi in np.round(E_pi_k, 2)])+ ']$', loc='left')
plt.legend()
plt.show()

#%%

## 推論処理

# 試行回数を指定
MaxIter = 50

# 変数を作成
eta_nk = np.zeros((N, K))

# 推移の確認用の受け皿を作成
trace_s_ink = [E_s_nk]
trace_a_ik = [list(a_hat_k)]
trace_b_ik = [list(b_hat_k)]
trace_alpha_ik = [list(alpha_k)]

# 変分推論
for i in range(MaxIter):
    # 潜在変数の事後分布のパラメータの各項を計算:式(4.60-4.62)
    E_lmd_k = a_hat_k / b_hat_k
    E_ln_lmd_k = psi(a_hat_k) - np.log(b_hat_k)
    E_ln_pi_k = psi(alpha_hat_k) - psi(np.sum(alpha_hat_k))
    
    for n in range(N):
        # 潜在変数の事後分布のパラメータを計算:式(4.51)
        tmp_eta_k = np.exp(x_n[n] * E_ln_lmd_k - E_lmd_k + E_ln_pi_k)
        eta_nk[n] = tmp_eta_k / np.sum(tmp_eta_k) # 正規化
        
        # 潜在変数の事後分布の期待値を計算:式(4.59)
        E_s_nk[n] = eta_nk[n].copy()
    
    # lambdaの事後分布のパラメータを計算:式(4.55)
    a_hat_k = np.sum(E_s_nk.T * x_n, axis=1) + a
    b_hat_k = np.sum(E_s_nk, axis=0) + b
    
    # 初期値によるpiのパラメータを計算:式(4.58)
    alpha_hat_k = np.sum(E_s_nk, axis=0) + alpha_k
    
    # 値を記録
    trace_s_ink.append(E_s_nk)
    trace_a_ik.append(list(a_hat_k))
    trace_b_ik.append(list(b_hat_k))
    trace_alpha_ik.append(list(alpha_hat_k))
    
    # 動作確認
    print(str(i + 1) + ' (' + str(np.round((i + 1) / MaxIter * 100, 1)) + '%)')

#%%

## パラメータの事後分布を確認

# 作図用のlambdaの点を作成
lambda_line = np.linspace(0, 2 * np.max(lambda_truth_k), num=1000)

# lambdaの事後分布を計算
posterior_lambda_kl = np.empty((K, len(lambda_line)))
for k in range(K):
    posterior_lambda_kl[k] = gamma.pdf(x=lambda_line, a=a_hat_k[k], scale=1 / b_hat_k[k])

# lambdaの事後分布を作図
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.vlines(x=lambda_truth_k[k], ymin=0.0, ymax=np.max(posterior_lambda_kl), 
               color='red', linestyle='--') # 真の値
    plt.plot(lambda_line, posterior_lambda_kl[k], label='cluster:' + str(k + 1)) # 事後分布
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', size=20)
plt.title('$iter:' + str(MaxIter) + ', N=' + str(N) + 
          ', \hat{a}=[' + ', '.join([str(a) for a in np.round(a_hat_k, 1)]) + ']' + 
          ', \hat{b}=[' + ', '.join([str(b) for b in np.round(b_hat_k, 1)]) + ']$', loc='left')
plt.legend()
plt.show()

#%%

# lambdaの平均値を計算:式(2.59)
E_lambda_k = a_hat_k / b_hat_k

# piの平均値を計算:式(2.51)
E_pi_k = alpha_hat_k / np.sum(alpha_hat_k)

# 最後の推定値による混合分布を計算
res_prob = 0.0
for k in range(K):
    # クラスタkの分布の確率を計算
    tmp_prob = poisson.pmf(k=x_line, mu=E_lambda_k[k])
    
    # K個の分布の加重平均を計算
    res_prob += tmp_prob * E_pi_k[k]

# 最後の推定値による分布を作図
plt.figure(figsize=(12, 9))
plt.bar(x=x_line, height=model_prob, label='truth model', 
        color='white', alpha=1, edgecolor='red', linestyle='--') # 真の分布
plt.bar(x=x_line, height=res_prob, color='purple') # 最後の推定値による分布
plt.xlabel('x')
plt.ylabel('prob')
plt.suptitle('Poisson Mixture Model:Variational Inference', size = 20)
plt.title('$iter:' + str(MaxIter) + ', N=' + str(N) + 
          ', E[\lambda]=[' + ', '.join([str(lmd) for lmd in np.round(E_lambda_k, 2)]) + ']' + 
          ', E[\pi]=[' + ', '.join([str(pi) for pi in np.round(E_pi_k, 2)])+ ']$', loc='left')
plt.legend()
plt.show()

#%%

# 確率が最大のクラスタ番号を抽出
s_n = np.argmax(E_s_nk, axis=1)


# 作図用の潜在変数の事後分布の期待値を計算:式(4.59)
E_s_lk = np.empty((len(x_line), K))
for n in range(len(x_line)):
    # 潜在変数の事後分布のパラメータを計算:式(4.51)
    tmp_eta_k = np.exp(x_line[n] * E_ln_lmd_k - E_lmd_k + E_ln_pi_k)
    E_s_lk[n] = tmp_eta_k / np.sum(tmp_eta_k) # 正規化

# 作図用のxのクラスタとなる確率を抽出
E_s_line = E_s_lk[np.arange(len(x_line)), np.argmax(E_s_lk, axis=1)]


# K個の色を指定
colormap_list = ['Reds', 'Greens', 'Blues']
color_list = ['red', 'green', 'blue']

# 最後のクラスタのヒストグラムを作成
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.bar(x=x_line, height=[np.sum(x_n[s_truth_n == k] == x) for x in x_line], 
            color='white', alpha=1, edgecolor=color_list[k], linestyle='--', label='true:' + str(k + 1)) # 真のクラスタ
for k in range(K):
    # k番目のグラフのカラーマップを設定
    cm = plt.get_cmap(colormap_list[k])
    plt.bar(x=x_line, height=[np.sum(x_n[s_n == k] == x) for x in x_line], 
            color=[cm(p) for p in E_s_line], alpha=1, label='cluster:' + str(k + 1)) # 最後のクラスタ
plt.xlabel('x')
plt.ylabel('count')
plt.suptitle('Poisson Mixture Model:Variational Inference', size=20)
plt.title('$iter:' + str(MaxIter) + ', N=' + str(N) + 
          ', E[\lambda]=[' + ', '.join([str(lmd) for lmd in np.round(a_hat_k / b_hat_k, 2)]) + ']' + 
          ', E[\pi]=[' + ', '.join([str(pi) for pi in np.round(alpha_hat_k / np.sum(alpha_hat_k), 2)]) + ']$', 
          loc='left')
plt.legend()
plt.show()

#%%

## 超パラメータの推移の確認

# aの推移を作図
plt.figure(figsize=(12, 9))
for k  in range(K):
    plt.plot(np.arange(MaxIter + 1), np.array(trace_a_ik).T[k], label='cluster:' + str(k + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Variational Inference', size=20)
plt.title('$\hat{a}$', loc='left')
plt.legend()
plt.grid()
plt.show()

#%%

# bの推移を作図
plt.figure(figsize=(12, 9))
for k  in range(K):
    plt.plot(np.arange(MaxIter + 1), np.array(trace_b_ik).T[k], label='cluster:' + str(k + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Variational Inference', size=20)
plt.title('$\hat{b}$', loc='left')
plt.legend()
plt.grid()
plt.show()

#%%

# alphaの推移を作図
plt.figure(figsize=(12, 9))
for k  in range(K):
    plt.plot(np.arange(MaxIter + 1), np.array(trace_alpha_ik).T[k], label='cluster:' + str(k + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Variational Inference', size=20)
plt.title('$\hat{\\alpha}$', loc='left')
plt.legend()
plt.grid()
plt.show()

#%%

## アニメーションによる確認

# 追加ライブラリ
import matplotlib.animation as animation

#%%

## 事後分布の推移の確認

# 作図用のlambdaの点を作成
lambda_line = np.linspace(0, 2 * np.max(lambda_truth_k), num=1000)

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_posterior(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のlambdaの事後分布を計算
    posterior_lambda_k = np.empty((K, len(lambda_line)))
    for k in range(K):
        posterior_lambda_k[k] = gamma.pdf(x=lambda_line, a=trace_a_ik[i][k], scale=1 / trace_b_ik[i][k])
    
    # i回目のlambdaの事後分布を作図
    for k in range(K):
        plt.vlines(x=lambda_truth_k[k], ymin=0.0, ymax=np.max(posterior_lambda_k), 
                   color='red', linestyle='--') # 真の値
        plt.plot(lambda_line, posterior_lambda_k[k], label='cluster:' + str(k + 1)) # 事後分布
    plt.xlabel('$\lambda$')
    plt.ylabel('density')
    plt.suptitle('Variational Inference', size=20)
    plt.title('$iter:' + str(i) + ', N=' + str(N) + 
              ', \hat{a}=[' + ', '.join([str(a) for a in np.round(trace_a_ik[i], 1)]) + ']' + 
              ', \hat{b}=[' + ', '.join([str(b) for b in np.round(trace_b_ik[i], 1)]) + ']$', loc='left')
    plt.legend()

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=MaxIter + 1, interval=100)
posterior_anime.save("ch4_3_3_Posterior.gif")

#%%

## サンプルの推移の確認

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_model(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のパラメータの平均値を計算:式(2.59,2.51)
    E_lambda_k = np.array(trace_a_ik[i]) / np.array(trace_b_ik[i])
    E_pi_k = np.array(trace_alpha_ik[i]) / np.sum(trace_alpha_ik [i])
    
    # i回目の推定値による混合分布を計算
    res_prob = 0.0
    for k in range(K):
        # クラスタkの分布の確率を計算
        tmp_prob = poisson.pmf(k=x_line, mu=E_lambda_k[k])
        
        # K個の分布の加重平均を計算
        res_prob += tmp_prob * E_pi_k[k]
    
    # i回目のサンプルによる分布を作図
    plt.bar(x=x_line, height=model_prob, label='true model', 
            color='white', alpha=1, edgecolor='red', linestyle='--') # 真の分布
    plt.bar(x=x_line, height=res_prob, color='purple') # 推定値による分布
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.suptitle('Poisson Mixture Model:Gibbs Sampling', size = 20)
    plt.title('$iter:' + str(i) + ', N=' + str(N) + 
              ', E[\lambda]=[' + ', '.join([str(lmd) for lmd in np.round(E_lambda_k, 2)]) + ']' + 
              ', E[\pi]=[' + ', '.join([str(pi) for pi in np.round(E_pi_k, 2)]) + ']$', loc='left')
    plt.ylim(0.0, 0.1)
    plt.legend()

# gif画像を作成
model_anime = animation.FuncAnimation(fig, update_model, frames=MaxIter + 1, interval=100)
model_anime.save("ch4_3_3_Model.gif")

#%%

# K個の色を指定
colormap_list = ['Reds', 'Greens', 'Blues']
color_list = ['red', 'green', 'blue']

# 作図用のxの点を作成
x_line = np.arange(0, 2 * np.max(lambda_truth_k))

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_cluster(i):
    # 超パラメータを取得
    a_hat_k = np.array(trace_a_ik[i])
    b_hat_k = np.array(trace_b_ik[i])
    alpha_hat_k = np.array(trace_alpha_ik[i])
    
    # 潜在変数の事後分布のパラメータの各項を計算:式(4.60-4.62)
    E_lmd_k = a_hat_k / b_hat_k
    E_ln_lmd_k = psi(a_hat_k) - np.log(b_hat_k)
    E_ln_pi_k = psi(alpha_hat_k) - psi(np.sum(alpha_hat_k))
    
    # 作図用の潜在変数の事後分布の期待値を計算:式(4.59)
    E_s_lk = np.empty((len(x_line), K))
    for n in range(len(x_line)):
        # 潜在変数の事後分布のパラメータを計算:式(4.51)
        tmp_eta_k = np.exp(x_line[n] * E_ln_lmd_k - E_lmd_k + E_ln_pi_k)
        E_s_lk[n] = tmp_eta_k / np.sum(tmp_eta_k) # 正規化
    
    # 作図用のxのクラスタとなる確率を抽出
    E_s_line = E_s_lk[np.arange(len(x_line)), np.argmax(E_s_lk, axis=1)]
    
    # 確率が最大のクラスタ番号を抽出
    s_n = np.argmax(trace_s_ink[i], axis=1)
    
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のクラスタの散布図を作成
    for k in range(K):
        plt.bar(x=x_line, height=[np.sum(x_n[s_truth_n == k] == x) for x in x_line], 
                color='white', alpha=1, edgecolor=color_list[k], linestyle='--', label='true:' + str(k + 1)) # 真のクラスタ
    for k in range(K):
        # k番目のグラフのカラーマップを設定
        cm = plt.get_cmap(colormap_list[k])
        plt.bar(x=x_line, height=[np.sum(x_n[s_n == k] == x) for x in x_line], 
                color=[cm(p) for p in E_s_line], alpha=1, label='cluster:' + str(k + 1)) # 推定したクラスタ
    plt.xlabel('x')
    plt.ylabel('count')
    plt.suptitle('Poisson Mixture Model', size=20)
    plt.title('$iter:' + str(i) + ', N=' + str(N) + 
              ', E[\lambda]=[' + ', '.join([str(lmd) for lmd in np.round(a_hat_k / b_hat_k, 2)]) + ']' + 
              ', E[\pi]=[' + ', '.join([str(pi) for pi in np.round(alpha_hat_k / np.sum(alpha_hat_k), 2)]) + ']$', loc='left')
    plt.legend()

# gif画像を作成
cluster_anime = animation.FuncAnimation(fig, update_cluster, frames=MaxIter + 1, interval=100)
cluster_anime.save("ch4_3_3_Cluster.gif")


#%%

# 確率が最大のクラスタ番号を抽出
s_n = np.argmax(E_s_nk, axis=1)


# 作図用の潜在変数の事後分布の期待値を計算:式(4.59)
E_s_lk = np.empty((len(x_line), K))
for n in range(len(x_line)):
    # 潜在変数の事後分布のパラメータを計算:式(4.51)
    tmp_eta_k = np.exp(x_line[n] * E_ln_lmd_k - E_lmd_k + E_ln_pi_k)
    E_s_lk[n] = tmp_eta_k / np.sum(tmp_eta_k) # 正規化

# 作図用のxのクラスタとなる確率を抽出
E_s_line = E_s_lk[np.arange(len(x_line)), np.argmax(E_s_lk, axis=1)]



# 最後のクラスタのヒストグラムを作成
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.bar(x=x_line, height=[np.sum(x_n[s_truth_n == k] == x) for x in x_line], 
            color='white', alpha=1, edgecolor=color_list[k], linestyle='--', label='true:' + str(k + 1)) # 真のクラスタ
for k in range(K):
    # k番目のグラフのカラーマップを設定
    cm = plt.get_cmap(colormap_list[k])
    plt.bar(x=x_line, height=[np.sum(x_n[s_n == k] == x) for x in x_line], 
            color=[cm(p) for p in E_s_line], alpha=1, label='cluster:' + str(k + 1)) # 最後のクラスタ
plt.xlabel('x')
plt.ylabel('count')
plt.suptitle('Poisson Mixture Model:Variational Inference', size=20)
plt.title('$iter:' + str(MaxIter) + ', N=' + str(N) + 
          ', E[\lambda]=[' + ', '.join([str(lmd) for lmd in np.round(a_hat_k / b_hat_k, 2)]) + ']' + 
          ', E[\pi]=[' + ', '.join([str(pi) for pi in np.round(alpha_hat_k / np.sum(alpha_hat_k), 2)]) + ']$', 
          loc='left')
plt.legend()
plt.show()


#%%

print('end')

