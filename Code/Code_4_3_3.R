
# 4.3.3 ポアソン混合モデルにおける変分推論 -------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)


# 真の観測モデルの設定 ----------------------------------------------------------------

# (観測)データ数を指定
N <- 100

# 真のパラメータを指定
lambda_truth <- c(5, 25)
pi_truth <- c(0.3, 0.7)

# クラスタ数
K <- length(lambda_truth)

# クラスタ(潜在変数)
s_nk <- rmultinom(n =  N, size = 1, prob = pi_truth) %>% 
  t()

# (観測)データXを生成
x_n <- rpois(n = N, lambda = apply(lambda_truth ^ t(s_nk), 2, prod))

# (観測)データを確認
summary(x_n)
tibble(
  x = x_n
) %>% 
  ggplot(aes(x = x)) + 
    geom_bar(fill = "#56256E") + 
    labs(title = "Histogram")


# パラメータの設定 ----------------------------------------------------------------

# 試行回数
Iter <- 50

# ハイパーパラメータa,bを指定
a <- 1
b <- 1

# lambda_kの期待値を計算:式(4.60),(4.61)
E_lambda_k <- rep(a, K) / rep(b, K)
E_ln_lambda_k <- digamma(rep(a, K)) - log(rep(a, K))

# ハイパーパラメータalphaを指定
alpha_k <- rep(2, K)

# piの期待値を計算:式(4.62)
E_ln_pi_k <- digamma(alpha_k) - digamma(sum(alpha_k))

# 変分推論 --------------------------------------------------------------------

# 受け皿を用意
eta_nk <- matrix(0, nrow = N, ncol = K)
E_s_nk <- matrix(0, nrow = N, ncol = K)
hat_a_k <- rep(0, K)
hat_b_k <- rep(0, K)

# 推移の確認用データフレームを作成
trace_parameter_df <- tibble(
  hat_a = rep(a, K), 
  hat_b = rep(b, K), 
  hat_alpha = hat_alpha_k, 
  cluster = as.factor(1:K), 
  Iteration = 0 # 初期値
)
trace_alpha <- matrix(0, nrow = K, ncol = Iter + 1)
trace_alpha[, 1] <- alpha_k
for(i in 1:Iter) {
  
  for(n in 1:N) {
    # パラメータを計算:式(4.51)
    tmp_eta <- exp(x_n[n] * E_ln_lambda_k - E_lambda_k + E_ln_pi_k)
    eta_nk[n, ] <- tmp_eta / sum(tmp_eta)
    
    # q(s_n)を更新:式(4.50)
    
    # s_nkの期待値を計算
    E_s_nk[n, ] <- eta_nk[n, ] * x_n[n]
  }
  
  for(k in 1:K) {
    
    # パラメータを計算:式(4.55)
    hat_a_k[k] <- sum(E_s_nk[, k] * x_n) + a
    hat_b_k[k] <- sum(E_s_nk[, k]) + b
    
    # q(lambda_k)を更新:式(4.54)
    
    
    # lambda_kの期待値を計算:式(4.60),(4.61)
    E_lambda_k[k] <- hat_a_k[k] / hat_b_k[k]
    E_ln_lambda_k[k] <- digamma(hat_a_k[k]) - log(hat_b_k[k])
  }
  
  # パラメータを計算:式(4.58)
  hat_alpha_k <- apply(E_s_nk, 2, sum) + alpha_k
  
  # q(pi)を更新:式(4.57)
  ln_C_D <- lgamma(sum(hat_alpha_k)) - lgamma(hat_alpha_k[k])
  exp(ln_C_D + (hat_alpha_k - 1) * log(pi_k))
  pi_k <- 1 / K
  # pi_kの期待値を計算:式(4.62)
  E_ln_pi_k <- digamma(hat_alpha_k) - digamma(sum(hat_alpha_k))
  
  # 推移の確認用データフレームを作成
  tmp_parameter_df <- data.frame(
    hat_a = hat_a_k, 
    hat_b = hat_b_k, 
    hat_alpha = hat_alpha_k, 
    cluster = as.factor(1:K), 
    Iteration = i # 試行回数
  )
  trace_alpha[, i + 1] <- hat_alpha_k
  # 結合
  trace_parameter_df <- rbind(trace_parameter_df, tmp_parameter_df)
}

# 結果の確認 -------------------------------------------------------------------

trace_alpha_wide <- cbind(
  as.data.frame(trace_alpha), 
  cluster = as.factor(1:K)
)
trace_alpha_long <- pivot_longer(
  trace_alpha_wide, 
  cols = -cluster, 
  names_to = "Iteration", 
  names_prefix = "V", 
  names_ptypes = list(Iteration = numeric()), 
  values_to = "value"
)

ggplot(trace_alpha_long, aes(Iteration, value, color = cluster)) + 
  geom_line()
