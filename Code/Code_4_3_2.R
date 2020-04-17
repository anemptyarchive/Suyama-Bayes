
# 4.3.2 ポアソン混合モデルにおけるギブスサンプリング --------------------------------------------------

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
tibble(
  x = x_n
) %>% 
  ggplot(aes(x = x)) + 
    geom_bar(fill = "#56256E")


# パラメータの設定 ----------------------------------------------------------------

# 試行回数を指定
Iter <- 50

# ハイパーパラメータa,bを指定
a <- 1
b <- 1

# パラメータlambdaを生成
lambda_k <- rgamma(n = K, shape = a, rate = b)

# ハイパーパラメータalphaを指定
alpha_k <- rep(2, K)

# パラメータpiを生成
pi_k <- MCMCpack::rdirichlet(n = 1, alpha = alpha_k) %>% 
  as.vector()


# ギブスサンプリング ---------------------------------------------------------------

# 受け皿を用意
eta_nk <- matrix(0, nrow = N, ncol = K)
a_hat_k <- rep(0, K)
b_hat_k <- rep(0, K)

# 推移の確認用データフレームを作成
trace_parameter_df <- tibble(
  a_hat = rep(a, K), 
  b_hat = rep(b, K), 
  alpha_hat = alpha_k, 
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
    alpha_hat = alpha_hat_k, 
    cluster = as.factor(1:K), 
    Iteration = i # 試行回数
  )
  # 結合
  trace_parameter_df <- rbind(trace_parameter_df, tmp_parameter_df)
}


# 推定結果の確認 -------------------------------------------------------------------

## lambdaの事後分布
# 作図用のデータフレームを作成
posterior_lambda_df <- tibble()
for(k in 1:K) {
  tmp_lambda_df <- tibble(
    x = seq(0, max(x_n), by = 0.01), 
    density = dgamma(x, shape = a_hat_k[k], rate = b_hat_k[k]), 
    cluster = as.factor(k)
  )
  posterior_lambda_df <- rbind(posterior_lambda_df, tmp_lambda_df)
}

# 作図
ggplot(posterior_lambda_df, aes(x, density, color = cluster)) + 
  geom_line() + # 折れ線グラフ
  scale_color_manual(values = c("#00A968", "orange")) + # グラフの色(不必要)
  geom_vline(xintercept = lambda_truth, color = "pink", linetype = "dashed") + # 垂直線
  labs(title = "Poisson mixture model:Gibbs sampling", 
       subtitle = paste0("a_hat=(", paste0(a_hat_k, collapse = ", "), 
                         "), b_hat=(", paste0(b_hat_k, collapse = ", "), ")")) # ラベル


## piの事後分布(K=2の場合)
# 作図用のデータフレームを作成
posterior_pi_df <- tibble()
for(k in 1:K) {
  tmp_pi_df <- tibble(
    x = seq(0, 1, by = 0.01), 
    density = dbeta(x, shape1 = alpha_hat_k[k], shape2 = alpha_hat_k[2 / k]), 
    cluster = as.factor(k)
  )
  posterior_pi_df <- rbind(posterior_pi_df, tmp_pi_df)
}

# 作図
ggplot(posterior_pi_df, aes(x, density, color = cluster)) + 
  geom_line() + # 折れ線グラフ
  scale_color_manual(values = c("#00A968", "orange")) + # グラフの色(不必要)
  geom_vline(xintercept = pi_truth, color = "pink", linetype = "dashed") + # 垂直線
  labs(title = "Poisson mixture model:Gibbs sampling", 
       subtitle = paste0("alpha_hat=(", paste0(alpha_hat_k, collapse = ", "), ")")) # ラベル


# 推移の確認 -------------------------------------------------------------------

### パラメータの事後分布の推移
## lambdaの事後分布の推移
# 作図用のデータフレームを作成
trace_lambda_df <- tibble()
for(i in 1:(Iter + 1)) {
  for(k in 1:K) {
    tmp_parameter <- trace_parameter_df %>% 
      filter(Iteration == i - 1, cluster == k)
    tmp_lambda_df <- tibble(
      x = seq(0, max(x_n), by = 0.001), 
      density = dgamma(x, shape = tmp_parameter[["a_hat"]], rate = tmp_parameter[["b_hat"]]), 
      cluster = tmp_parameter[["cluster"]], 
      Iteration = tmp_parameter[["Iteration"]]
    )
    trace_lambda_df <- rbind(trace_lambda_df, tmp_lambda_df)
  }
}

# 作図
trace_lambda_graph <- ggplot(trace_lambda_df, aes(x, density, color = cluster)) + 
  geom_line() + # 折れ線グラフ
  scale_color_manual(values = c("#00A968", "orange")) + # グラフの色(不必要)
  geom_vline(xintercept = lambda_truth, 
             color = "pink", linetype = "dashed") + # 垂直線
  transition_manual(Iteration) + # フレーム
  labs(title = "Poisson mixture model:Gibbs sampling", 
       subtitle = "i={current_frame}") # ラベル

# 描画
animate(trace_lambda_graph, nframes = Iter + 1, fps = 10)


## piの事後分布の推移
# 作図用のデータフレームを作成
trace_pi_df <- tibble()
for(i in 1:(Iter + 1)) {
  for(k in 1:K) {
    tmp_parameter <- trace_parameter_df %>% 
      filter(Iteration == i - 1)
    tmp_pi_df <- tibble(
      x = seq(0, 1, by = 0.01), 
      density = dbeta(x, shape1 = tmp_parameter[["alpha_hat"]][k], shape2 = tmp_parameter[["alpha_hat"]][2 / k]), 
      cluster = as.factor(k), 
      Iteration = i - 1
    )
    trace_pi_df <- rbind(trace_pi_df, tmp_pi_df)
  }
}

# 作図
trace_pi_graph <- ggplot(trace_pi_df, aes(x, density, color = cluster)) + 
  geom_line() + # 折れ線グラフ
  scale_color_manual(values = c("#00A968", "orange")) + # グラフの色(不必要)
  geom_vline(xintercept = pi_truth, 
             color = "pink", linetype = "dashed") + # 垂直線
  transition_manual(Iteration) + # フレーム
  labs(title = "Poisson mixture model:Gibbs sampling", 
       subtitle = "i={current_frame}") # ラベル

# 描画
animate(trace_pi_graph, nframes = Iter + 1, fps = 10)


### ハイパーパラメータの推移
## 全表示
# 作図
trace_parameter_df %>% 
  pivot_longer(
    cols = -c(cluster, Iteration), 
    names_to = "parameter", 
    names_ptypes = list(parameter = factor()), 
    values_to = "value"
  ) %>% 
  mutate(parameters = paste(parameter, cluster, sep = "_")) %>% 
  ggplot(aes(Iteration, value, color = parameters)) + 
    geom_line() + # 垂直線
    labs(title = "Poisson mixture model:Gibbs sampling") # ラベル


## クラスタを指定してグラフ化
# クラスタ番号kを指定
cluster_num <- 1

# 作図
trace_parameter_df %>% 
  pivot_longer(
    cols = -c(cluster, Iteration), 
    names_to = "parameter", 
    names_ptypes = list(parameter = factor()), 
    values_to = "value"
  ) %>% 
  filter(cluster == cluster_num) %>% 
  ggplot(aes(Iteration, value, color = parameter)) + 
    geom_line() + # 垂直線
    labs(title = "Poisson mixture model:Gibbs sampling", 
         subtitle = paste0("k=", cluster_num)) # ラベル


