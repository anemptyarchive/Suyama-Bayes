
# 1次元ガウスモデル --------------------------------------------------------------

# chapter 3.3.3
# 平均と精度が未知の場合
# ベイズ推論
# 推論アルゴリズムの実装


#%%

# 3.3.3項で利用するライブラリ
import numpy as np
from scipy.special import gammaln # 対数ガンマ関数
from scipy.stats import norm, gamma, t # 1次元ガウス分布, ガンマ分布, 1次元スチューデントのt分布
import matplotlib.pyplot as plt

#%%

## 尤度(ガウス分布)の設定

# 真の平均パラメータを指定
mu_truth = 25.0

# 真の精度パラメータを指定
lambda_truth = 0.01
print(np.sqrt(1.0 / lambda_truth)) # 標準偏差


# 作図用のxの値を作成
x_line = np.linspace(
    mu_truth - 4.0 * np.sqrt(1.0 / lambda_truth), 
    mu_truth + 4.0 * np.sqrt(1.0 / lambda_truth), 
    num=1000
)

# 尤度の確率密度を計算:式(2.64)
ln_C_N = - 0.5 *(np.log(2.0 * np.pi) - np.log(lambda_truth)) # 正規化項(対数)
model_dens = np.exp(ln_C_N - 0.5 * lambda_truth * (x_line - mu_truth)**2)
#model_dens = norm.pdf(x=x_line, loc=mu_truth, scale=np.sqrt(1 / lambda_truth))

#%%

# 尤度を作図
plt.figure(figsize=(12, 9))
plt.plot(x_line, model_dens, label='true model') # 真の分布
plt.xlabel('x')
plt.ylabel('density')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$\mu=' + str(mu_truth) + ', \lambda=' + str(lambda_truth) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 観測データの生成

# (観測)データ数を指定
N = 50

# ガウス分布に従うデータを生成
x_n = np.random.normal(loc=mu_truth, scale=np.sqrt(1.0 / lambda_truth), size=N)
print(x_n[:5])

#%%

# 観測データのヒストグラムを作図
plt.figure(figsize=(12, 9))
#plt.hist(x=x_n, bins=50, label='data') # 観測データ:(度数)
plt.hist(x=x_n, density=True, bins=50, label='data') # 観測データ:(相対度数)
plt.plot(x_line, model_dens, color='red', linestyle='--', label='true model') # 真の分布
plt.xlabel('x')
plt.ylabel('count')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \mu=' + str(mu_truth) + ', \lambda=' + str(lambda_truth) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 事前分布(ガウス・ガンマ分布)の設定

# muの事前分布のパラメータを指定
m = 0.0
beta = 1.0

# lambdaの事前分布のパラメータを指定
a = 1.0
b = 1.0


# 作図用のmuの値を作成
mu_line = np.linspace(mu_truth - 30, mu_truth + 30, num=1000)


# lambdaの期待値を計算:式(2.59)
E_lambda = a / b

# muの事前分布の確率密度を計算:式(2.64)
ln_C_N = - 0.5 * (np.log(2.0 * np.pi) - np.log(beta * E_lambda)) # 正規化項(対数)
prior_mu_dens = np.exp(ln_C_N - 0.5 * beta * E_lambda * (mu_line - m)**2)
#prior_mu_dens = norm.pdf(x=mu_line, loc=m, scale=np.sqrt(1.0 / beta / E_lambda)) # 確率密度


# 作図用のlambdaの値を作成
lambda_line = np.linspace(0.0, 4.0 * lambda_truth, num=1000)

# lambdaの事前分布の確率密度を計算:式(2.56)
ln_C_Gam = a * np.log(b) - gammaln(a) # 正規化項(対数)
prior_lambda_dens = np.exp(ln_C_Gam + (a - 1.0) * np.log(lambda_line) - b * lambda_line)
#prior_lambda_dens = gamma.pdf(x=lambda_line, a=a, scale=1.0 / b)

#%%

# muの事前分布を作図
plt.figure(figsize=(12, 9))
plt.plot(mu_line, prior_mu_dens, color='purple', label='$\mu$ prior') # muの事前分布
plt.vlines(x=mu_truth, ymin=0.0, ymax=np.nanmax(prior_mu_dens), 
           color='red', linestyle='--', label='true val') # 真の値
plt.xlabel('$\mu$')
plt.ylabel('density')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$m=' + str(m) + 
          ', \\beta=' + str(beta) + 
          ', E[\lambda]=' + str(np.round(E_lambda, 5)) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# lambdaの事前分布を作図
plt.figure(figsize=(12, 9))
plt.plot(lambda_line, prior_lambda_dens, color='purple', label='$\lambda$ prior') # lambdaの事前分布
plt.vlines(x=lambda_truth, ymin=0.0, ymax=np.nanmax(prior_lambda_dens) * 2.0, 
           color='red', linestyle='--', label='true val') # 真の値
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$a=' + str(a) + ', b=' + str(b) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 事後分布(ガウス・ガンマ分布)の計算

# muの事後分布のパラメータを計算:式(3.83)
beta_hat = N + beta
m_hat = (np.sum(x_n) + beta * m) / beta_hat


# lambdaの事後分布のパラメータを計算:式(3.88)
a_hat = 0.5 * N + a
b_hat = 0.5 * (np.sum(x_n**2) + beta * m**2 - beta_hat * m_hat**2) + b


# lambdaの期待値を計算:式(2.59)
E_lambda_hat = a_hat / b_hat

# muの事後分布の確率密度を計算:式(2.64)
ln_C_N = - 0.5 * (np.log(2.0 * np.pi) - np.log(beta_hat * E_lambda_hat)) # 正規化項(対数)
posterior_mu_dens = np.exp(ln_C_N - 0.5 * beta_hat * E_lambda_hat * (mu_line - m_hat)**2)
#posterior_mu_dens = norm.pdf(x=mu_line, loc=m_hat, scale=np.sqrt(1.0 / beta_hat / E_lambda_hat)) # 確率密度


# lambdaの事後分布の確率密度を計算:式(2.56)
ln_C_Gam = a_hat * np.log(b_hat) - gammaln(a_hat) # 正規化項(対数)
posterior_lambda_dens = np.exp(
    ln_C_Gam + (a_hat - 1.0) * np.log(lambda_line) - b_hat * lambda_line
)
#posterior_lambda_dens = gamma.pdf(x=lambda_line, a=a_hat, scale=1.0 / b_hat)

#%%

# muの事後分布を作図
plt.figure(figsize=(12, 9))
plt.plot(mu_line, posterior_mu_dens, color='purple', label='$\mu$ prior') # muの事後分布
plt.vlines(x=mu_truth, ymin=0.0, ymax=np.nanmax(posterior_mu_dens), 
           color='red', linestyle='--', label='true val') # 真の値
plt.xlabel('$\mu$')
plt.ylabel('density')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{m}=' + str(np.round(m_hat, 1)) + 
          ', \hat{\\beta}=' + str(beta_hat) + 
          ', E[\hat{\lambda}]=' + str(np.round(E_lambda_hat, 5)) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# lambdaの事後分布を作図
plt.figure(figsize=(12, 9))
plt.plot(lambda_line, posterior_lambda_dens, color='purple', label='$\lambda$ posterior') # lambdaの事後分布
plt.vlines(x=lambda_truth, ymin=0.0, ymax=np.nanmax(posterior_lambda_dens), 
           color='red', linestyle='--', label='true val') # 真の値
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', a=' + str(a_hat) + ', b=' + str(np.round(b_hat, 1)) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 予測分布(スチューデントのt分布)の計算

# 予測分布のパラメータを計算:式(3.95')
mu_st_hat = m_hat
lambda_st_hat = beta_hat * a_hat / (1.0 + beta_hat) / b_hat
nu_st_hat = 2.0 * a_hat
mu_st_hat = (np.sum(x_n) + beta_hat * m) / (N + beta)
#numer_lambda = (N + beta) * (0.5 * N + a)
#denom_lambda = (N + 1 + beta) * (0.5 * (np.sum(x_n**2) + beta + m**2 - beta_hat * m_hat**2) + beta)
#lambda_s_hat = numer_lambda / denom_lambda
#nu_s_hat = N + 2 * a


# 予測分布の確率密度を計算:式(3.76)
ln_C_St = gammaln(0.5 * (nu_st_hat + 1.0)) - gammaln(0.5 * nu_st_hat) # 正規化項(対数)
ln_term1 = 0.5 * np.log(lambda_st_hat / np.pi / nu_st_hat)
ln_term2 = - 0.5 * (nu_st_hat + 1.0) * np.log(1.0 + lambda_st_hat / nu_st_hat * (x_line - mu_st_hat)**2)
predict_dens = np.exp(ln_C_St + ln_term1 + ln_term2)
#predict_dens = t.pdf(x=x_line, df=nu_st_hat, loc=mu_st_hat, scale=np.sqrt(1.0 / lambda_st_hat))

#%%

# 予測分布を作図
plt.figure(figsize=(12, 9))
plt.plot(x_line, predict_dens, color='purple', label='predict') # 予測分布
plt.plot(x_line, model_dens, color='red', linestyle='--', label='true model') # 真の分布
plt.xlabel('x')
plt.ylabel('density')
plt.suptitle("Student's t Distribution", fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{\mu}_s=' + str(np.round(mu_st_hat, 1)) + 
          ', \hat{\lambda}_s=' + str(np.round(lambda_st_hat, 5)) + 
          ', \hat{\\nu}_s=' + str(nu_st_hat) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()


#%%
