
# 4.3.2 ポアソン混合モデルにおけるギブスサンプリング --------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(MCMCpack) # rdirichlet()のため


# パラメータの設定 ----------------------------------------------------------------

# 観測データ数を指定
N <- 20

# 真のパラメータを指定
lambda_true <- c(1, 2, 3, 3.5)
K <- length(lambda_true)

# 観測データXを生成
x_n <- rpois(n = N, lambda = lambda_true)

# 試行回数を指定
Iter <- 10

# ハイパーパラメータa,bを指定
a <- 2
b <- 0.5

# パラメータlambdaを生成
lambda_k <- rgamma(n = K, shape = a, rate = b)

# ハイパーパラメータalphaを指定
alpha_k <- rep(2, K)

# パラメータpiを生成
pi_k <- MCMCpack::rdirichlet(n = 1, alpha = alpha_k) %>% 
  as.vector()

# 潜在変数Sの初期値
s_nk <- matrix(0, nrow = N, ncol = K)

# 受け皿
eta_nk <- matrix(0, nrow = N, ncol = K)


# ギブスサンプリング ---------------------------------------------------------------

# 推移の確認用データフレームを作成
trace_df <- data.frame(
  alpha = alpha_k, 
  a = rep(a, K), 
  b = rep(b, K), 
  pi = pi_k, 
  k = as.factor(1:K), 
  Iter = 0
)

for(i in 1:Iter) {
  
  for(n in 1:N) {
    
    # ハイパーパラメータを更新
    tmp_eta_k <- exp(x_n[n] * log(lambda_k) - lambda_k + log(pi_k))
    eta_nk[n, ] <- tmp_eta_k <- sum(tmp_eta_k)
    
    # 潜在変数sをサンプリング:式(4.37)
    s_nk[n, ] <- rmultinom(n =  1, size = 1, prob = eta_nk[n, ]) %>% 
      as.vector()
  }
  
  for(k in 1:K) {
    
    # ハイパーパラメータを更新
    a_hat_k <- sum(s_nk[, k] * x_n) + a
    b_hat_k <- sum(s_nk[, k]) + b
    
    # パラメータlambdaをサンプリング:式(4.41)
    lambda_k <- rgamma(n = K, shape = a_hat_k, rate = b_hat_k)
  }
  
  # ハイパーパラメータをalpha更新:式(4.45)
  alpha_hat_k <- apply(s_nk, 2, sum) + alpha_k
  
  # パラメータpiをサンプリング:式(4.44)
  pi_k <- MCMCpack::rdirichlet(n = 1, alpha = alpha_hat_k) %>% 
    as.vector()
  
  # 推移の確認用データフレームを作成
  tmp_trace_df <- data.frame(
    alpha = alpha_hat_k, 
    a = a_hat_k, 
    b = b_hat_k, 
    pi = pi_k, 
    k = as.factor(1:K), 
    Iter = i
  )
  # 結合
  trace_df <- rbind(trace_df, tmp_trace_df)
}


# 結果の確認 -------------------------------------------------------------------
