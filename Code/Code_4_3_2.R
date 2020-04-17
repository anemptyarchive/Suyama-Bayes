
# 4.3.2 ポアソン混合モデルにおけるギブスサンプリング --------------------------------------------------

# 利用パッケージ
library(tidyverse)
library(gganimate)


# パラメータの設定 ----------------------------------------------------------------

# 真のパラメータを指定
lambda_true <- c(5, 25)

# 観測データ数を指定
N <- 100

# 観測データXを生成
x_n <- rpois(n = N, lambda = lambda_true)

# 観測データを確認
tibble(
  x = x_n
) %>% 
  ggplot(aes(x = x)) + 
    geom_bar(fill = "#56256E")

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
trace_parameter_df <- tibble(
  a_hat = rep(a, K), 
  b_hat = rep(b, K), 
  cluster = as.factor(1:K), 
  Iteration = 0 # 初期値
)

for(i in 1:Iter) {
  
  for(n in 1:N) {
    
    # ハイパーパラメータeta_nを計算:式(4.38)
    tmp_eta_k <- exp(x_n[n] * log(lambda_k) - lambda_k + log(pi_k))
    eta_nk[n, ] <- tmp_eta_k / sum(tmp_eta_k) # 正規化
    
    # 潜在変数s_nをサンプリング:式(4.37)
    s_nk[n, ] <- rmultinom(n =  1, size = 1, prob = eta_nk[n, ]) %>% 
      as.vector()
  }
  
  for(k in 1:K) {
    
    # ハイパーパラメータhat{a}_k,hat{b}_kを計算:式(4.42)
    a_hat_k[k] <- sum(s_nk[, k] * x_n) + a
    b_hat_k[k] <- sum(s_nk[, k]) + b
    
    # パラメータlambda_kをサンプリング:式(4.41)
    lambda_k <- rgamma(n = K, shape = a_hat_k, rate = b_hat_k)
  }
  
  # ハイパーパラメータをhat{alpha}を計算:式(4.45)
  alpha_hat_k <- apply(s_nk, 2, sum) + alpha_k
  
  # パラメータpiをサンプリング:式(4.44)
  pi_k <- MCMCpack::rdirichlet(n = 1, alpha = alpha_hat_k) %>% 
    as.vector()
  
  # 推移の確認用データフレームを作成
  tmp_parameter_df <- data.frame(
    a_hat = a_hat_k, 
    b_hat = b_hat_k, 
    cluster = as.factor(1:K), 
    Iteration = i # 試行回数
  )
  # 結合
  trace_parameter_df <- rbind(trace_parameter_df, tmp_parameter_df)
}


# 結果の確認 -------------------------------------------------------------------

posterior_df <- tibble()
for(k in 1:K) {
  tmp_df <- tibble(
    x = seq(0, max(x_n), by = 0.01), 
    density = dgamma(x, shape = a_hat_k[k], rate = b_hat_k[k]), 
    cluster = as.factor(k)
  )
  posterior_df <- rbind(posterior_df, tmp_df)
}

# 作図
ggplot(posterior_df, aes(x, density, color = cluster)) + 
  geom_line() + # 折れ線グラフ
  scale_color_manual(values = c("#00A968", "orange")) + # グラフの色(不必要)
  geom_vline(xintercept = lambda_true, color = "pink", linetype = "dashed") + # 垂直線
  labs(title = "Poisson mixture model:Gibbs sampling", 
       subtitle = paste0("a_hat=(", paste0(a_hat_k, collapse = ", "), 
                         "), b_hat=(", paste0(b_hat_k, collapse = ", "), ")")) # ラベル


# 推移の確認 -------------------------------------------------------------------

## 事後分布の推移
# 作図用のデータフレームを作成
trace_posterior_df <- tibble()
for(i in 1:(Iter + 1)) {
  for(k in 1:K) {
    tmp_parameter <- trace_parameter_df %>% 
      filter(Iteration == i - 1, cluster == k)
    tmp_posterior <- tibble(
      x = seq(0, max(x_n), by = 0.01), 
      density = dgamma(x, shape = tmp_parameter[["a_hat"]], rate = tmp_parameter[["b_hat"]]), 
      cluster = tmp_parameter[["cluster"]], 
      Iteration = tmp_parameter[["Iteration"]]
    )
    trace_posterior_df <- rbind(trace_posterior_df, tmp_posterior)
  }
}

# 作図
posterior_graph <- ggplot(trace_posterior_df, aes(x, density, color = cluster)) + 
  geom_line() + # 折れ線グラフ
  scale_color_manual(values = c("#00A968", "orange")) + # グラフの色(不必要)
  geom_vline(xintercept = lambda_true, 
             color = "pink", linetype = "dashed") + # 垂直線
  transition_manual(Iteration) + # フレーム
  labs(title = "Poisson mixture model:Gibbs sampling", 
       subtitle = "i={current_frame}") # ラベル

# 描画
animate(posterior_graph, nframes = Iter + 1, fps = 10)


## ハイパーパラメータa,bの推移
# クラスタ番号kを指定
coluster_num <- 1

# 作図
trace_parameter_df %>% 
  pivot_longer(
    cols = -c(cluster, Iteration), 
    names_to = "parameter", 
    names_ptypes = list(parameter = factor()), 
    values_to = "value"
  ) %>% 
  filter(cluster == coluster_num) %>% 
  ggplot(aes(Iteration, value, color = parameter)) + 
    geom_line() + # 垂直線
    labs(title = "Poisson mixture model:Gibbs sampling") # ラベル


