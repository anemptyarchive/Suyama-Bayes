
# 4.3.2 ポアソン混合モデルにおけるギブスサンプリング --------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)


# パラメータの設定 ----------------------------------------------------------------

# 観測データ数を指定
N <- 100

# 真のパラメータを指定
lambda_true <- c(5, 25)

# 観測データXを生成
x_n <- rpois(n = N, lambda = lambda_true)

# 観測データを確認
tibble(
  x = x_n
) %>% 
  ggplot(aes(x = x)) + 
    geom_bar()

# 試行回数を指定
Iter <- 50

# クラスタ数
K <- length(lambda_true)

# ハイパーパラメータa,bを指定
a <- 2
b <- 2

# パラメータlambdaを生成
lambda_k <- rgamma(n = K, shape = a, rate = b)

# ハイパーパラメータalphaを指定
alpha_k <- rep(2, K)

# パラメータpiを生成
pi_k <- MCMCpack::rdirichlet(n = 1, alpha = alpha_k) %>% 
  as.vector()

# 潜在変数Sの初期値
s_nk <- matrix(0, nrow = N, ncol = K)


# ギブスサンプリング ---------------------------------------------------------------

# 受け皿を用意
eta_nk <- matrix(0, nrow = N, ncol = K)
a_hat_k <- rep(0, K)
b_hat_k <- rep(0, K)

# 推移の確認用データフレームを作成
parameter_df <- tibble(
  a_hat = rep(a, K), 
  b_hat = rep(b, K), 
  alpha_hat = alpha_k, 
  k = as.factor(1:K), 
  Iter = 0
)
posterior_df <- tibble(
  x = rep(seq(0, max(x_n), by = 0.01), each = K), 
  value = dgamma(x, shape = rep(a, K), rate = rep(b, K)), 
  Iter = 0
)

for(i in 1:Iter) {
  
  for(n in 1:N) {
    
    # ハイパーパラメータを更新
    tmp_eta_k <- exp(x_n[n] * log(lambda_k) - lambda_k + log(pi_k))
    eta_nk[n, ] <- tmp_eta_k / sum(tmp_eta_k)
    
    # 潜在変数sをサンプリング:式(4.37)
    s_nk[n, ] <- rmultinom(n =  1, size = 1, prob = eta_nk[n, ]) %>% 
      as.vector()
  }
  
  for(k in 1:K) {
    
    # ハイパーパラメータを更新
    a_hat_k[k] <- sum(s_nk[, k] * x_n) + a
    b_hat_k[k] <- sum(s_nk[, k]) + b
    
    # パラメータlambdaをサンプリング:式(4.41)
    lambda_k <- rgamma(n = K, shape = a_hat_k, rate = b_hat_k)
  }
  
  # ハイパーパラメータをalpha更新:式(4.45)
  alpha_hat_k <- apply(s_nk, 2, sum) + alpha_k
  
  # パラメータpiをサンプリング:式(4.44)
  pi_k <- MCMCpack::rdirichlet(n = 1, alpha = alpha_hat_k) %>% 
    as.vector()
  
  # 推移の確認用データフレームを作成
  tmp_parameter_df <- data.frame(
    a_hat = a_hat_k, 
    b_hat = b_hat_k, 
    alpha_hat = alpha_hat_k, 
    k = as.factor(1:K), 
    Iter = i
  )
  tmp_posterior_df <- tibble(
    x = rep(seq(0, max(x_n), by = 0.01), each = K), 
    value = dgamma(x, shape = a_hat_k, rate = b_hat_k), 
    Iter = i
  )
  # 結合
  parameter_df <- rbind(parameter_df, tmp_parameter_df)
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
}


# 結果の確認 -------------------------------------------------------------------

posterior_df <- tibble()
for(k in 1:K) {
  tmp_df <- tibble(
    x = seq(0, max(x_n), by = 0.1), 
    value = dgamma(x, shape = a_hat_k[k], rate = b_hat_k[k]), 
    k = as.factor(k)
  )
  posterior_df <- rbind(posterior_df, tmp_df)
}

# 作図
ggplot(posterior_df, aes(x = x, y = value, color = k)) + 
  geom_line() + 
  geom_vline(xintercept = lambda_true, color = "orange", linetype = "dashed") + 
  labs(title = "Poisson mixture model:Gibbs sampling", 
       subtitle = expression(lambda))

# 推移の確認 -------------------------------------------------------------------

# 作図
posterior_graph <- ggplot(posterior_df, aes(x, value)) + 
  geom_line(color = "#00A968") + 
  geom_vline(xintercept = lambda_true, 
             color = "orange", linetype = "dashed") + 
  transition_manual(Iter) + 
  labs(title = "Poisson mixture model:Gibbs sampling", 
       subtitle = "i={current_frame}")

# 描画
animate(posterior_graph, nframes = Iter + 1, fps = 5)

col_name
parameter_df %>% 
  select(-alpha_hat) %>% 
  pivot_longer(
    cols = -c(k, Iter), 
    names_to = "parameter", 
    names_ptypes = list(parameter = factor()), 
    values_to = "value"
  ) %>% 
  ggplot(aes(parameter, value, color = k)) + 
    geom_line()
