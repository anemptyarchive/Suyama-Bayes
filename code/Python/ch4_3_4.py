# 4.3.4 ポアソン混合分布における推論：崩壊型ギブスサンプリング

#%%

# 4.3.4項で利用するライブラリ
import numpy as np
from scipy.stats import poisson, gamma, nbinom # ポアソン分布, ガンマ分布, 負の二項分布
import matplotlib.pyplot as plt

#%%

## 観測モデル(ポアソン混合分布)の設定

# K個の真のパラメータを指定
lambda_truth_k  = np.array([10.0, 25.0, 40.0])

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
    model_prob += pi_truth_k[k] * tmp_prob

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


# 作図用のlambdaの点を作成
lambda_line = np.linspace(0.0, 2.0 * np.max(lambda_truth_k), num=1000)

#%%

# lambdaの事前分布を計算
prior_lambda = gamma.pdf(x=lambda_line, a=a, scale=1 / b)

# lambdaの事前分布を作図
plt.figure(figsize=(12, 9))
plt.plot(lambda_line, prior_lambda, label='prior', color='purple') # 事前分布
plt.vlines(x=lambda_truth_k, ymin=0.0, ymax=np.max(prior_lambda), label='true val', 
           color='red', linestyle='--') # 真の値
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('a=' + str(a) + ', b=' + str(b), loc='left')
plt.legend()
plt.show()

#%%

## 初期値の設定

# 潜在変数の初期値をランダムに生成
s_nk = np.random.multinomial(n=1, pvals=alpha_k / np.sum(alpha_k), size=N)
print(s_nk[:5])

# 初期値によるlambdaの事後分布のパラメータを計算:式(4.24)
a_hat_k = np.sum(s_nk.T * x_n, axis=1) + a
b_hat_k = np.sum(s_nk, axis=0) + b
print(a_hat_k)
print(b_hat_k)

# 初期値によるpiの事後分布のパラメータを計算:式(4.45)
alpha_hat_k = np.sum(s_nk, axis=0) + alpha_k
print(alpha_hat_k)

#%%

# 初期値によるlambdaの近似事後分布を作図
posterior_lambda_kl = np.empty((K, len(lambda_line)))
for k in range(K):
    posterior_lambda_kl[k] = gamma.pdf(x=lambda_line, a=a_hat_k[k], scale=1 / b_hat_k[k])

# 初期値によるlambdaの近似事後分布を作図
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.plot(lambda_line, posterior_lambda_kl[k], label='cluster:' + str(k + 1)) # 事後分布
plt.vlines(x=lambda_truth_k, ymin=0.0, ymax=np.max(posterior_lambda_kl), 
           color='red', linestyle='--', label='true val') # 真の値
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', size=20)
plt.title('$iter:' + str(0) + ', N=' + str(N) + 
          ', \hat{a}=[' + ', '.join([str(a) for a in a_hat_k]) + ']' + 
          ', \hat{b}=[' + ', '.join([str(b) for b in b_hat_k]) + ']$', loc='left')
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
    init_prob += E_pi_k[k] * tmp_prob

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
MaxIter = 100

# 推移の確認用の受け皿を作成
_, s_n = np.where(s_nk == 1) # クラスタ番号を抽出
trace_s_in = [list(s_n)]
trace_a_ik = [list(a_hat_k)]
trace_b_ik = [list(b_hat_k)]
trace_alpha_ik = [list(alpha_k)]

# 崩壊型ギブスサンプリング
for i in range(MaxIter):
    for n in range(N):
        # n番目のデータの現在のクラスタ番号を取得
        k, = np.where(s_nk[n] == 1)[0]
        
        # n番目のデータに関する統計量を除算
        a_hat_k[k] = a_hat_k[k] - x_n[n]
        b_hat_k[k] = b_hat_k[k] - 1
        alpha_hat_k[k] = alpha_hat_k[k] - 1
        
        # 負の二項分布(4.81)のパラメータを計算
        r_hat_k = a_hat_k
        p_hat_k = 1 / (b_hat_k + 1)
        
        # 負の二項分布の確率を計算:式(4.81)
        prob_nb_k = nbinom.pmf(k=x_n[n], n=r_hat_k, p=1 - p_hat_k)
        
        # カテゴリ分布(4.74)のパラメータを計算:式(4.75)
        eta_k = alpha_hat_k / np.sum(alpha_hat_k)
        
        # n番目のクラスタのサンプリング確率を計算:式(4.66)
        prob_s_k = (prob_nb_k + 1e-7) * eta_k
        prob_s_k = prob_s_k / np.sum(prob_s_k) # 正規化
        
        # n番目のクラスタをサンプル:式(4.74)
        s_nk[n] = np.random.multinomial(n=1, pvals=prob_s_k, size=1)
        
        # n番目のデータの新しいクラスタを取得
        k, = np.where(s_nk[n] == 1)[0]
        
        # n番目のデータに関する統計量を加算:式(4.82-4.83)
        a_hat_k[k] = a_hat_k[k] + x_n[n]
        b_hat_k[k] = b_hat_k[k] + 1
        alpha_hat_k[k] = alpha_hat_k[k] + 1
    
    # 値を記録
    _, s_n = np.where(s_nk == 1) # クラスタ番号を抽出
    trace_s_in.append(list(s_n))
    trace_a_ik.append(list(a_hat_k))
    trace_b_ik.append(list(b_hat_k))
    trace_alpha_ik.append(list(alpha_hat_k))
    
    # 動作確認
    print(str(i + 1) + ' (' + str(np.round((i + 1) / MaxIter * 100, 1)) + '%)')


#%%

## パラメータの事後分布を確認

# lambdaの事後分布を計算
posterior_lambda_kl = np.empty((K, len(lambda_line)))
for k in range(K):
    posterior_lambda_kl[k] = gamma.pdf(x=lambda_line, a=a_hat_k[k], scale=1 / b_hat_k[k])

# lambdaの事後分布を作図
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.plot(lambda_line, posterior_lambda_kl[k], label='cluster:' + str(k + 1)) # 事後分布
plt.vlines(x=lambda_truth_k, ymin=0.0, ymax=np.max(posterior_lambda_kl), 
           color='red', linestyle='--', label='true val') # 真の値
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution:Collapsed Gibbs Sampling', size=20)
plt.title('$iter:' + str(MaxIter) + ', N=' + str(N) + 
          ', \hat{a}=[' + ', '.join([str(a) for a in a_hat_k]) + ']' + 
          ', \hat{b}=[' + ', '.join([str(b) for b in b_hat_k]) + ']$', loc='left')
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
    res_prob += E_pi_k[k] * tmp_prob

# 最後の推定値による分布を作図
plt.figure(figsize=(12, 9))
plt.bar(x=x_line, height=model_prob, label='truth model', 
        color='white', alpha=1, edgecolor='red', linestyle='--') # 真の分布
plt.bar(x=x_line, height=res_prob, color='purple') # 最後の推定値による分布
plt.xlabel('x')
plt.ylabel('prob')
plt.suptitle('Poisson Mixture Model:Collapsed Gibbs Sampling', size = 20)
plt.title('$iter:' + str(MaxIter) + ', N=' + str(N) + 
          ', E[\lambda]=[' + ', '.join([str(lmd) for lmd in np.round(E_lambda_k, 2)]) + ']' + 
          ', E[\pi]=[' + ', '.join([str(pi) for pi in np.round(E_pi_k, 2)])+ ']$', loc='left')
plt.legend()
plt.show()

#%%

# K個の色を指定
color_list = ['red', 'green', 'blue']

# 最後のクラスタのヒストグラムを作成
plt.figure(figsize=(12, 9))
for k in range(K):
    plt.bar(x=x_line, height=[np.sum(x_n[s_truth_n == k] == x) for x in x_line], 
            color='white', alpha=1, edgecolor=color_list[k], linestyle='--', label='true cluster:' + str(k + 1)) # 真のクラスタ
for k in range(K):
    plt.bar(x=x_line, height=[np.sum(x_n[s_n == k] == x) for x in x_line], 
            alpha=0.5, label='cluster:' + str(k + 1)) # 最後のクラスタ
plt.xlabel('x')
plt.ylabel('count')
plt.suptitle('Poisson Mixture Model', size=20)
plt.title('$N=' + str(N) + 
          ', E[\lambda]=[' + ', '.join([str(lmd) for lmd in np.round(a_hat_k / b_hat_k, 2)]) + ']' + 
          ', E[\pi]=[' + ', '.join([str(pi) for pi in np.round(alpha_hat_k / np.sum(alpha_hat_k), 2)]) + ']$', loc='left')
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
plt.suptitle('Collapsed Gibbs Sampling', size=20)
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
plt.suptitle('Collapsed Gibbs Sampling', size=20)
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
plt.suptitle('Collapsed Gibbs Sampling', size=20)
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
lambda_line = np.linspace(0.0, 2.0 * np.max(lambda_truth_k), num=1000)

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
    plt.suptitle('Gamma Distribution:Collapsed Gibbs Sampling', size=20)
    plt.title('$iter:' + str(i) + ', N=' + str(N) + 
              ', \hat{a}=[' + ', '.join([str(a) for a in trace_a_ik[i]]) + ']' + 
              ', \hat{b}=[' + ', '.join([str(b) for b in trace_b_ik[i]]) + ']$', loc='left')
    plt.legend()
    plt.grid()

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=MaxIter + 1, interval=100)
posterior_anime.save("ch4_3_4_Posterior.gif")

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
        res_prob += E_pi_k[k] * tmp_prob
    
    # i回目のサンプルによる分布を作図
    plt.bar(x=x_line, height=model_prob, label='true model', 
            color='white', alpha=1, edgecolor='red', linestyle='--') # 真の分布
    plt.bar(x=x_line, height=res_prob, color='purple') # 推定値による分布
    plt.xlabel('x')
    plt.ylabel('prob')
    plt.suptitle('Poisson Mixture Model:Collapsed Gibbs Sampling', size = 20)
    plt.title('$iter:' + str(i) + ', N=' + str(N) + 
              ', E[\lambda]=[' + ', '.join([str(lmd) for lmd in np.round(E_lambda_k, 2)]) + ']' + 
              ', E[\pi]=[' + ', '.join([str(pi) for pi in np.round(E_pi_k, 2)]) + ']$', loc='left')
    plt.ylim(0.0, 0.1)
    plt.legend()
    plt.grid()

# gif画像を作成
model_anime = animation.FuncAnimation(fig, update_model, frames=MaxIter + 1, interval=100)
model_anime.save("ch4_3_4_Model.gif")

#%%

# K個の色を指定
color_list = ['red', 'green', 'blue']

# 作図用のxの点を作成
x_line = np.arange(0, 2 * np.max(lambda_truth_k))

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_cluster(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のクラスタの散布図を作成
    for k in range(K):
        plt.bar(x=x_line, height=[np.sum(x_n[s_truth_n == k] == x) for x in x_line], 
                color='white', alpha=1, edgecolor=color_list[k], linestyle='--', label='true cluster:' + str(k + 1)) # 真のクラスタ
    for k in range(K):
        plt.bar(x=x_line, height=[np.sum(x_n[np.array(trace_s_in[i]) == k] == x) for x in x_line], 
                alpha=0.5, label='cluster:' + str(k + 1)) # サンプルしたクラスタ
    plt.xlabel('x')
    plt.ylabel('count')
    plt.suptitle('Poisson Mixture Model:Collapsed Gibbs Sampling', size=20)
    plt.title('$iter:' + str(i) + ', N=' + str(N) + 
              ', \lambda=[' + ', '.join([str(lmd) for lmd in np.round(np.array(trace_a_ik[i]) / np.array(trace_b_ik[i]), 2)]) + ']' + 
              ', \pi=[' + ', '.join([str(pi) for pi in np.round(np.array(trace_alpha_ik[i]) / np.sum(trace_alpha_ik [i]), 2)]) + ']$', loc='left')
    plt.legend()
    plt.grid()

# gif画像を作成
cluster_anime = animation.FuncAnimation(fig, update_cluster, frames=MaxIter + 1, interval=100)
cluster_anime.save("ch4_3_4_Cluster.gif")


#%%

print('end')

