
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

# クラスタ(潜在変数)を生成
s_nk <- rmultinom(n =  N, size = 1, prob = pi_truth) %>% 
  t()

# (観測)データXを生成
x_n <- rpois(n = N, lambda = apply(lambda_truth^t(s_nk), 2, prod))

# 観測データを確認
summary(x_n)
tibble(x = x_n) %>% 
  ggplot(aes(x = x)) + 
    geom_bar(fill = "#56256E") + 
    labs(title = "Histogram")


# パラメータの設定 ----------------------------------------------------------------

# 試行回数を指定
MaxIter <- 50

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
hat_a_k <- rep(0, K)
hat_b_k <- rep(0, K)

# ハイパーパラメータの推定値の推移の確認用
trace_a <- matrix(0, nrow = MaxIter + 1, ncol = K)
trace_b <- matrix(0, nrow = MaxIter + 1, ncol = K)
trace_alpha <- matrix(0, nrow = MaxIter + 1, ncol = K)
# 初期値を代入
trace_a[1, ] <- a
trace_b[1, ] <- b
trace_alpha[1, ] <- alpha_k

for(i in 1:MaxIter) {
  
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
    hat_a_k[k] <- sum(s_nk[, k] * x_n) + a
    hat_b_k[k] <- sum(s_nk[, k]) + b
    
    # パラメータlambda_kをサンプリング:式(4.41)
    lambda_k <- rgamma(n = K, shape = hat_a_k, rate = hat_b_k)
  }
  
  # ハイパーパラメータをhat{alpha}を計算:式(4.45)
  hat_alpha_k <- apply(s_nk, 2, sum) + alpha_k
  
  # パラメータpiをサンプリング:式(4.44)
  pi_k <- MCMCpack::rdirichlet(n = 1, alpha = hat_alpha_k) %>% 
    as.vector()
  
  # 推移の確認用に推定結果を保存
  trace_a[i + 1, ] <- hat_a_k
  trace_b[i + 1, ] <- hat_b_k
  trace_alpha[i + 1, ] <- hat_alpha_k
}


# 推定結果の確認 -------------------------------------------------------------------

## lambdaの事後分布
# 作図用のデータフレームを作成
posterior_lambda_df <- tibble()
for(k in 1:K) {
  tmp_lambda_df <- tibble(
    x = seq(0, max(x_n), by = 0.01), 
    density = dgamma(x, shape = hat_a_k[k], rate = hat_b_k[k]), 
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
       subtitle = paste0("a_hat=(", paste0(hat_a_k, collapse = ", "), 
                         "), b_hat=(", paste0(hat_b_k, collapse = ", "), ")")) # ラベル


## piの事後分布(K=2の場合)
# 作図用のデータフレームを作成
posterior_pi_df <- tibble()
for(k in 1:K) {
  tmp_pi_df <- tibble(
    x = seq(0, 1, by = 0.01), 
    density = dbeta(x, shape1 = hat_alpha_k[k], shape2 = hat_alpha_k[2 / k]), 
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
       subtitle = paste0("alpha_hat=(", paste0(hat_alpha_k, collapse = ", "), ")")) # ラベル


# パラメータの事後分布の推移の確認 -------------------------------------------------------------------

## lambdaの近似事後分布
# 作図用のデータフレームを作成
trace_lambda_long <- tibble()
for(i in 1:(MaxIter + 1)) {
  for(k in 1:K) {
    # データフレームに変換
    tmp_lambda_df <- tibble(
      lambda = seq(0, max(x_n), by = 0.01), 
      density = dgamma(lambda, shape = trace_a[i, k], rate = trace_b[i, k]), 
      cluster = as.factor(k), 
      Iteration = i - 1
    )
    # 結合
    trace_lambda_long <- rbind(trace_lambda_long, tmp_lambda_df)
  }
}

# 作図
graph_lambda <- ggplot(trace_lambda_long, aes(lambda, density, color = cluster)) + 
  geom_line() + # 折れ線グラフ
  scale_color_manual(values = c("#00A968", "orange")) + # グラフの色(不必要)
  geom_vline(xintercept = lambda_truth, color = "pink", linetype = "dashed") + # 垂直線
  transition_manual(Iteration) + # フレーム
  labs(title = "Poisson Mixture Model:Gibbs Sampling", 
       subtitle = "i={current_frame}") # ラベル

# gif画像を作成
animate(graph_lambda, nframes = MaxIter + 1, fps = 5)


## piの近似事後分布(K=2のときのみ可)
# 作図用のデータフレームを作成
trace_pi_long <- tibble()
for(i in 1:(MaxIter + 1)) {
  for(k in 1:K) {
    # データフレームに変換
    tmp_pi_df <- tibble(
      pi = seq(0, 1, by = 0.001), 
      density = dbeta(pi, shape1 = trace_alpha[i, k], shape2 = trace_alpha[i, 2 / k]), 
      cluster = as.factor(k), 
      Iteration = i - 1
    )
    # 結合
    trace_pi_long <- rbind(trace_pi_long, tmp_pi_df)
  }
}

# 作図
graph_pi <- ggplot(trace_pi_long, aes(pi, density, color = cluster)) + 
  geom_line() + # 折れ線グラフ
  scale_color_manual(values = c("#00A968", "orange")) + # グラフの色(不必要)
  geom_vline(xintercept = pi_truth, color = "pink", linetype = "dashed") + # 垂直線
  transition_manual(Iteration) + # フレーム
  labs(title = "Poisson Mixture Model:Gibbs Sampling", 
       subtitle = "i={current_frame}") # ラベル

# gif画像を作成
animate(graph_pi, nframes = MaxIter + 1, fps = 5)


# ハイパーパラメータの推移 ------------------------------------------------------------

## lambdaのパラメータa
# データフレームに変換
trace_a_wide <- cbind(
  as.data.frame(trace_a), 
  Iteration = 1:(MaxIter + 1)
)

# long型に変換
trace_a_long <- pivot_longer(
  trace_a_wide, 
  cols = -Iteration, 
  names_to = "cluster", 
  names_prefix = "V", 
  names_ptypes = list(cluster = factor()), 
  values_to = "value"
)

# 作図
ggplot(trace_a_long, aes(Iteration, value, color = cluster)) + 
  geom_line() + 
  labs(title = "Poisson Mixture Model:Gibbs Sampling", 
       subtitle = expression(hat(a)))


## lambdaのパラメータb
# データフレームに変換
trace_b_wide <- cbind(
  as.data.frame(trace_a), 
  Iteration = 1:(MaxIter + 1)
)

# long型に変換
trace_b_long <- pivot_longer(
  trace_b_wide, 
  cols = -Iteration, 
  names_to = "cluster", 
  names_prefix = "V", 
  names_ptypes = list(cluster = factor()), 
  values_to = "value"
)

# 作図
ggplot(trace_b_long, aes(Iteration, value, color = cluster)) + 
  geom_line() + 
  labs(title = "Poisson Mixture Model:Gibbs Sampling", 
       subtitle = expression(hat(b)))


## piのパラメータ
# データフレームに変換
trace_alpha_wide <- cbind(
  as.data.frame(trace_alpha), 
  Iteration = 1:(MaxIter + 1)
)

# long型に変換
trace_alpha_long <- pivot_longer(
  trace_alpha_wide, 
  cols = -Iteration, 
  names_to = "cluster", 
  names_prefix = "V", 
  names_ptypes = list(cluster = factor()), 
  values_to = "value"
)

# 作図
ggplot(trace_alpha_long, aes(Iteration, value, color = cluster)) + 
  geom_line() + 
  labs(title = "Poisson Mixture Model:Gibbs Sampling", 
       subtitle = expression(hat(alpha)))


